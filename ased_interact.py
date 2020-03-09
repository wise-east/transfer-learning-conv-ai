# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import os
import torch
import torch.nn.functional as F
import warnings
import re 

# Migration Notes: pytorch_pretrained_bert -> pytorch_transformers. 
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset_personalities, download_pretrained_model

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def predict_next_word(personality, history, tokenizer, model, args, current_output=None):

    # import pdb; pdb.set_trace()
    instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

    logits = model(input_ids, token_type_ids=token_type_ids)

    if isinstance(logits, tuple): 
        logits = logits[0]

    # logits = logits[0, -1, :] / args.temperature
    # migration notes: logits is a single value tuple. logits -> logits[0]
    logits = logits[0, -1, :] / args.temperature

    logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
    probs = F.softmax(logits, dim=-1)

    return probs 
 

def sample_sequence(personality, history, tokenizer, model, args):

    special_tokens = ['<bos>', '<eos>', '<speaker1>', '<speaker2>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    # get probabilities for first word
    probs = predict_next_word(personality, history, tokenizer, model, args, [])
    
    # sample the args.top_c number of samples
    prev_n = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, num_samples=args.top_c, replacement=True)
    
    outputs = [] 
    for i in range(len(prev_n)): 
        output = [] 
        for i in range(args.max_length):
            probs = predict_next_word(personality, history, tokenizer, model, args, output)
            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, num_samples=1)
            
            if i < args.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=args.top_c)

            if prev.item() in special_tokens_ids: 
                break 
            output.append(prev.item())

        outputs.append(output)

    return outputs

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", "-mc", type=str, default="runs/gpt2_convai_yesand", help="Path, url or short name of the model")
    parser.add_argument("--max_history", "-mh", type=int, default=5, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    # set a number of top choices to show
    parser.add_argument("--top_c", type=int, default=10, help="Determine how many top choices to be shown.")
    # add option to not use personality
    parser.add_argument("--no_personality", "-np", action='store_true', help="Set to not sample a personality.")
    # use text file to deduce results
    parser.add_argument("--email_sequence", "-es", default=None, help="Provide the text file for which to generate outputs for the chatbot.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if os.path.isdir("./huggingface_s3/"): 
            args.model_checkpoint = "./huggingface_s3/"
            logger.info("Loading from pre-downloaded temp path: {}".format(args.model_checkpoint))
        else: 
            args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)


    add_special_tokens_(model, tokenizer)
    model.to(args.device)
    model.eval()

    # added the option to opt out of using a personality 
    if args.no_personality: 
        logger.info("No personality is sampled for this chatbot.")
        personality = "" 
        # personality = ["My name is Isabelle Hawkins.", 
        #                "I am five years old.", 
        #                "My phone number is 959-100-9300.", 
        #                "Here is a link I would like you to check out: google.com.", 
        #                "I would like to know more about you."]
        # personality = [tokenizer.encode(p) for p in personality] 
        # logger.info("Selected custom personality: %s",tokenizer.decode(chain(*personality)))
    else:
        logger.info("Sample a personality")
        personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
        personality = random.choice(personalities)
        # import pdb; pdb.set_trace()
        logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))


    # test case to see if decoder works as expected. If not, see if special tokens are appropriately added to the tokenizer 
    test_case = "Hello, my name is Justin. I'd like a strawberry cheesecake."
    test_encode = tokenizer.encode(test_case)
    test_decode = tokenizer.decode(test_encode)
    assert test_case == test_decode

    # adapt code for efficient experimentation of existing email exchanges
    if args.email_sequence:
        logger.info(f"Running chatbot generations for {os.path.split(args.email_sequence)[-1]}")
        while True: 
            with open(args.email_sequence, 'r') as f: 
                email_sequence = f.readlines()
            his_length = int(input("Indicate how many exchanges you want to refer back to - must be an integer. \n1 indicates only the most recent email from the scammer: "))
            email_sequence = [e for e in email_sequence if re.sub('\n', '', e)] # remove any empty lines 
            email_sequence = email_sequence[-((his_length-1)*2+1):]
           
            history = [tokenizer.encode(e) for e in email_sequence]

            logger.info("Used input:\n")
            for idx, e in enumerate(email_sequence): 
                output = f"\tUser: {e}" if idx%2 else f"\tScammer: {e}"
                print(output)
            with torch.no_grad(): 
                out_ids = sample_sequence(personality, history, tokenizer, model, args) 

            out_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out_ids]

            print(f"Top {args.top_c} choices of history length = {his_length}:")
            for idx, o in enumerate(out_texts): 
                print(f"\t{idx}: {o}")

    # manual generation 
    else: 
        history = []
        while True: 
            custom_history = input("Press 0 to end\n\tAdd history: ")
            if custom_history == '0': 
                break 
            else: 
                history.append(tokenizer.encode(custom_history))

        while True:
            raw_text = input("Scammer >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Scammer >>> ")
            history.append(tokenizer.encode(raw_text))

            with torch.no_grad():
                out_ids = sample_sequence(personality, history, tokenizer, model, args)
            # multiple retries 
            # history.append(out_ids)
            # history = history[-(2*args.max_history+1):]
            out_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out_ids]
            print(f"Top {args.top_c} choices:")
            for idx, o in enumerate(out_texts): 
                print(f"\t{idx}: {o}")
    
if __name__ == "__main__":
    run()
