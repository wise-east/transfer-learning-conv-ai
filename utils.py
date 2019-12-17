# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import tarfile
from pathlib import Path 

import torch
from transformers import cached_path
from sklearn.model_selection import train_test_split


PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

""" Change this variable to the path for where your custom file is located""" 
CUSTOM_DATAPATH = "../../yes-and-data.json"
# optional variable for consistent splitting of data 
RANDOM_STATE = 42
MAX_LEN = 1024

logger = logging.getLogger(__file__)

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = "./tmp/"

    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def build_custom_input(path): 

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())

    reformatted_data = [] 

    if 'yesand' in path: 
        all_samples = [] 
        for k, v in dataset['yesands'].items(): 
            all_samples += v
        

        # formatting custom dataset in the same format as the ConvAI dataset used in original repo
        for idx, yesand, in enumerate(all_samples): 
            instance = {"personality": "", "utterances": []}
            utterance = {"history": [yesand['p']],
                            "candidates": [all_samples[(idx+1)%len(all_samples)]['r'], yesand['r']]}
            instance["utterances"].append(utterance)
            reformatted_data.append(instance)

    elif 'cornell' in path or 'dailydialog' in path or 'bolt' in path or 'question' in path: 
        reformatted_data = dataset 
    else: 
        logger.info("Unidentified custom dataset. Abort.", path)

    return reformatted_data

# ===================== TODO: Edit this part of the code to format your custom data for the pretrained model ===================== 

def get_custom_dataset(tokenizer, dataset_path, validset_path=None):
    """ Get custom data"""

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))[:100]
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    # get absolute file path of dataset and remove .json
    dataset_cache= Path(dataset_path).absolute().__str__().replace('.json', '')
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # For avoiding using GPT cache for GPT-2 and vice-versa

    if Path(dataset_cache).is_file():
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Load and tokenize dataset from %s", dataset_path)
        dataset = build_custom_input(dataset_path)
        if dataset: 
            dataset = tokenize(dataset)
            torch.save(dataset, dataset_cache)

    # optional loading of a separate validation set 
    if validset_path: 
        validset_cache= Path(validset_path).absolute().__str__().replace('.json', '')
        validset_cache = validset_cache + '_' + type(tokenizer).__name__  # For avoiding using GPT cache for GPT-2 and vice-versa

        if Path(validset_cache).is_file(): 
            logger.info("Load tokenized validation set from cache at %s", validset_cache)
            validset =torch.load(validset_cache)
        else: 
            logger.info("Load and tokenize validation set from %s", validset_path)
            validset = build_custom_input(validset_path)
            validset = tokenize(validset)
            torch.save(validset, validset_cache)

        dataset = {'train': dataset, 'valid': validset}

    # if there is no separate validation set, just split the dataset to train and valid 
    else: 
        train, valid = train_test_split(dataset, test_size=1000, random_state=RANDOM_STATE)
        dataset = {'train': train, 'valid': valid}

    return dataset

# ===================== END TODO =====================

def get_dataset(tokenizer, dataset_path, dataset_cache=None):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    if not dataset_cache:
        dataset_cache = 'personachat_cache_' + type(tokenizer).__name__  # For avoiding using GPT cache for GPT-2 and vice-versa

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        with open('persona_chat.json', 'w') as f:
            json.dump(obj=dataset['valid'], fp=f, indent=4, sort_keys=True)

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)

    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
