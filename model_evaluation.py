# model_evaluation.py:  The model evaluation process for BEC-Cri/Pro
#                       Can be modified for use with other datasets
#
# Author:               Mustafa Bozdag
# Date:                 04/28/2023

import math
import torch

from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from utils import input_pipeline, prob_with_prior

# Model evaluation function:
# - eval_df: Evaluation sentences BEC-Cri/Pro
# - tokenizer: Bert/AutoModel tokenizer
# - model: BertForMaskedLM/AutoModelForMaskedLM 
# - device: GPU/CPU
# - model and predicts the associations
# - Returns the association scores for all sentences
def model_evaluation(eval_df, tokenizer, model, device):

    # Choose the max_len as the smallest power of 2 greater or equal to the maximum sentence lenght
    max_len = max([len(sent.split()) for sent in eval_df.Sent_TM])
    pos = math.ceil(math.log2(max_len))
    max_len_eval = int(math.pow(2, pos))

    print('max_len evaluation: {}'.format(max_len_eval))

    # Create inputs for BEC-Cri/Pro for MLM model: 
    # Target masked, target, attribute masked, and the original tokenized inputs to recover the target word
    eval_tokens_TM, eval_attentions_TM = input_pipeline(eval_df.Sent_TM, tokenizer, max_len_eval)
    eval_tokens_TAM, eval_attentions_TAM = input_pipeline(eval_df.Sent_TAM, tokenizer, max_len_eval)
    eval_tokens, _ = input_pipeline(eval_df.Sentence, tokenizer, max_len_eval)

    # Check lengths before going further
    assert eval_tokens_TM.shape == eval_attentions_TM.shape
    assert eval_tokens_TAM.shape == eval_attentions_TAM.shape

    # Create an evaluation Dataloader
    eval_batch = 20
    eval_data = TensorDataset(eval_tokens_TM, eval_attentions_TM, eval_tokens_TAM, eval_attentions_TAM, eval_tokens)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch, sampler=eval_sampler)

    # Put everything to GPU (if it is available)
    eval_tokens_TM = eval_tokens_TM.to(device)
    eval_attentions_TM = eval_attentions_TM.to(device)
    eval_tokens_TAM = eval_tokens_TAM.to(device)
    eval_attentions_TAM = eval_attentions_TAM.to(device)
    model.to(device)

    # Put the model in evaluation mode
    model.eval()
    associations_all = []
    # Calculate the association scores
    for step, batch in enumerate(eval_dataloader):
        b_input_TM = batch[0].to(device)
        b_att_TM = batch[1].to(device)
        b_input_TAM = batch[2].to(device)
        b_att_TAM = batch[3].to(device)

        with torch.no_grad():
            outputs_TM = model(b_input_TM, attention_mask=b_att_TM)
            outputs_TAM = model(b_input_TAM, attention_mask=b_att_TAM)
            predictions_TM = softmax(outputs_TM[0], dim=2)
            predictions_TAM = softmax(outputs_TAM[0], dim=2)

        assert predictions_TM.shape == predictions_TAM.shape

        # Calculate the association scores
        associations = prob_with_prior(predictions_TM, predictions_TAM, b_input_TAM, batch[4], tokenizer)

        associations_all += associations

    return associations_all