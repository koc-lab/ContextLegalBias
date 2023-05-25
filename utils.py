# utils.py: Includes functions that are used throught the project to save space
#           and preserve the continuity of the process in other files
#
# Author:   Mustafa Bozdag
# Date:     04/28/2023

import time
import datetime
import numpy as np
import pandas as pd
import torch

from typing import List, Tuple
from keras.utils import pad_sequences
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# This class is a subclass of torch.utils.data.Dataset and it created a DocDataset object for LCD:
# - Returns DocDataset object to be passed to a DataLoader for fine-tuning with sequence classification
class DocDataset(Dataset):
    def __init__(self, path: str, max_seq_length: int, tokenizer) -> None:

        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sources, self.targets = self._load(path)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source: List[int] = self.sources[idx]
        seq_length = len(source)
        if seq_length > self.max_seq_length:
            input_ids = np.array(source[:self.max_seq_length])
            segment_ids = np.array([0] * self.max_seq_length)
            input_mask = np.array([1] * self.max_seq_length)
        else:
            # Zero-pad up to the sequence length
            pad = [0] * (self.max_seq_length - seq_length)
            input_ids = np.array(source + pad)  
            segment_ids = np.array([0] * self.max_seq_length)
            input_mask = np.array([1] * seq_length + pad)
        target = np.array(self.targets[idx])
        return input_ids, segment_ids, input_mask, target

    def _load(self, path: str) -> Tuple[List[List[int]], List[int]]:
        # Get the data from the given path of the .tsv file containing documents
        data_df = pd.read_csv(path, sep='\t')
        # Create empty arrays for source (token IDs) and targets (integer labels indicating applicant gender)
        sources, targets = [], []
        for i in range(len(data_df)):
            body = data_df['text'][i]
            target = int(data_df['applicant_gender'][i])-1
            targets.append(target)
            tokens: List[str] = ['[CLS]'] + self.tokenizer.tokenize(body) + ['[SEP]']
            sources.append(self.tokenizer.convert_tokens_to_ids(tokens))
        assert len(sources) == len(targets)
        return sources, targets

#Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):    
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed))) 
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
  

# From: https://github.com/allenai/dont-stop-pretraining/blob/master/scripts/mlm_study.py
# Prepare masked tokens inputs/labels: 80% MASK, 10% Random, 10% Original
def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[
    torch.Tensor, torch.Tensor]:
    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the "
                         "--mlm flag if you want to use this tokenizer. ")

    labels = inputs.clone()
    # Sample tokens in each sequence with 0.15 probability (default) for MLM training
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


# Provide the attention mask list of lists
# - 0 only for [PAD] tokens (index 0)
# - Returns torch tensor of attention masks
def attention_mask_creator(input_ids):
    attention_masks = []
    for sent in input_ids:
        segments_ids = [int(t > 0) for t in sent]
        attention_masks.append(segments_ids)
    return torch.tensor(attention_masks)

# Tokenize the sentences and map the tokens to their word IDs
def tokenize_to_id(sentences, tokenizer):
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   1: Tokenize the sentence.
        #   2: Prepend the `[CLS]` token to the start.
        #   3: Append the `[SEP]` token to the end.
        #   4: Map tokens to their IDs.
        encoded_sent = tokenizer.encode( sent, add_special_tokens=True)
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    return input_ids

# Tokenize, pad and create attention masks
def input_pipeline(sequence, tokenizer, MAX_LEN):
    input_ids = tokenize_to_id(sequence, tokenizer)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long",
                              value=tokenizer.mask_token_id,
                              truncating="post", padding="post")
    input_ids = torch.tensor(input_ids)

    attention_masks = attention_mask_creator(input_ids)

    return input_ids, attention_masks

# Calculate the association scores
def prob_with_prior(pred_TM, pred_TAM, input_ids_TAM, original_ids, tokenizer):
    pred_TM = pred_TM.cpu()
    pred_TAM = pred_TAM.cpu()
    input_ids_TAM = input_ids_TAM.cpu()

    probs = []
    for doc_idx, id_list in enumerate(input_ids_TAM):
        # see where the masks were placed in this sentence
        mask_indices = np.where(id_list == tokenizer.mask_token_id)[0]
        # now get the probability of the target word:
        # first get id of target word
        target_id = original_ids[doc_idx][mask_indices[0]]
        # get its probability with unmasked profession
        target_prob = pred_TM[doc_idx][mask_indices[0]][target_id].item()
        # get its prior probability (masked profession)
        prior = pred_TAM[doc_idx][mask_indices[0]][target_id].item()

        probs.append(np.log(target_prob / prior))

    return probs    