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

# Migration Notes: pytorch_pretrained_bert -> pytorch_transformers. 
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset_personalities, download_pretrained_model
from generate_valid import sample_sequence

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", "-mc", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    # add option to not use personality
    parser.add_argument("--no_personality", type=bool, default=True, help="Set to not sample a personality.")
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
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if "gpt2" == args.model else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    add_special_tokens_(model, tokenizer)
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

    history = []
    # while True: 
    #     custom_history = input("Press 0 to end\n\tAdd history: ")
    #     if custom_history == '0': 
    #         break 
    #     else: 
    #         history.append(tokenizer.encode(custom_history))

    while True: 
        history = [] 
        prompt = input("Speaker 1 >>> ")
        while not prompt:
            print('Prompt should not be empty!')
            prompt = input("Speaker 1 >>> ")
        history.append(tokenizer.encode(prompt))
        
        i = 0 
        while True:
            with torch.no_grad():
                out_ids = sample_sequence(personality, history, tokenizer, model, args)
            history.append(out_ids)
            history = history[-(2*args.max_history+1):]
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            i += 1
            speaker = "Speaker 2" if i%2 else "Speaker 1"
            print(f"{speaker}: {out_text}")

            if i == 10: 
                break
    

if __name__ == "__main__":
    run()
