# main.py:  The main file of the project
#
# Author:   Mustafa Bozdag
# Date:     04/28/2023

import os
import time
import random
import argparse
import copy

import math
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from nltk import sent_tokenize
from transformers import (BertTokenizer, 
                          BertForMaskedLM, 
                          BertForSequenceClassification, 
                          AutoTokenizer, 
                          AutoModelForMaskedLM, 
                          AutoModelForSequenceClassification)
                          
from utils import DocDataset, input_pipeline
from fine_tune import fine_tune
from model_evaluation import model_evaluation


# Parser for the arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Which BERT model to use (default: bert-base-uncased)', required=False, default='bert-base-uncased')
    parser.add_argument('--tune', help='.tsv file with sentences for fine-tuning (GAP-Flipped or ECtHR-GTuned)', required=False)
    parser.add_argument('--eval', help='.tsv file with sentences for bias evaluation (BEC-Pro or BEC-Cri)', required=True)
    parser.add_argument('--out', help='Output directory/Filename', required=False, default='')
    parser.add_argument('--batch', help='Fix batch-size for fine-tuning (default: 1)', required=False, default=1)
    parser.add_argument('--seed', required=False, default=42)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--eps', type=float, required=False, default=1e-8)
    parser.add_argument('--epoch', type=int, required=False, default=3)
    parser.add_argument('--devID', type=int, required=True)
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    
    # Get the arguments
    args = parse_arguments()
    # Chose the device according to the given argument
    devID = args.devID
    print("\n\n ----- RUNNING MB VER ------ \n\n")
    # Check GPU availibility and display used device ID 
    if torch.cuda.is_available():
        device = torch.device(type='cuda', index=devID)
        print('--- Using GPU with ID:', torch.cuda.get_device_name(devID))
    else:
        print('--- No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    # Set random seed for torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Show chosen evaluation dataset
    print('--- Preparing evaluation data: %s' % args.eval)
    # Import the evaluation data (.tsv file)
    eval_data = pd.read_csv(args.eval, sep='\t')


    # Import the model and corresponding tokenizer and set up the model for BERT
    print('--- Importing model: ', args.model)     
    if args.model == 'bert-base-uncased':
      tokenizer = BertTokenizer.from_pretrained(args.model)
      # Load the model as a Masked Language Model
      model = BertForMaskedLM.from_pretrained(args.model, output_attentions=False, output_hidden_states=False)   
    # Same process for other BERT-Based model types  
    else:
      tokenizer = AutoTokenizer.from_pretrained(args.model)
      model = AutoModelForMaskedLM.from_pretrained(args.model, output_attentions=False, output_hidden_states=False) 
                                              
    # Calculate the pre-association scores (pre meaning before fine-tuning) and display elapsed time
    print('\n--- Calculating pre-associations scores...')
    st = time.time()
    pre_associations = model_evaluation(eval_data, tokenizer, model, device)
    et = time.time()
    print('--- Calculation took {0:.2f} minutes'.format((et - st) / 60))
    # Add the associations to a dataframe to save
    eval_data = eval_data.assign(Pre_Assoc=pre_associations)   

    # Create a directory to save the debiased models if there isn't one
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Create a string for the model name (for saving) 
    model_str = args.model
    model_str = model_str.split('/')[-1]
    model_str = model_str.split('.')[0] 
        
    # Create a directory for results if there isn't one
    if not os.path.exists('results'):
        os.makedirs('results')  

    # Create a string for the evaluation dataset name (for saving)
    eval_str = args.eval
    eval_str = eval_str.split('/')[-1]
    eval_str = eval_str.split('.')[0]

    # Setting up the data for fine-tuning according to the dataset
    if args.tune:  
        print('--- Importing fine-tuning data: ', args.tune)
     
        if 'gap' in args.tune:
            # Read the data from the .tsv file
            tune_corpus = pd.read_csv(args.tune, sep='\t')
            tune_data = []
            # Append the tokenized text data to the array tune_data
            for text in tune_corpus.Text:
                tune_data += sent_tokenize(text)
    
            # Set the maximum sequence lenght as the smalles power of 2 greater than or equal to the maximum sentence length
            max_seq_length = max([len(sent.split()) for sent in tune_data])
            pos = math.ceil(math.log2(max_seq_length))
            max_seq_length = int(math.pow(2, pos))
            print('---Maximum sequence lenght for tuning: {}'.format(max_seq_length))
    
            # Get the tokens and the attentions tensor using "input_pipeline" from "utils.py"
            tune_tokens, tune_attentions = input_pipeline(tune_data, tokenizer, max_seq_length)
            # Continue if the dimensions of tokens and attentions are the same
            assert tune_tokens.shape == tune_attentions.shape
    
            # Create TendorDataset and RandomSampler objects to pass to the DataLoader
            train_dataset = TensorDataset(tune_tokens, tune_attentions)
            train_sampler = RandomSampler(train_dataset)
            # Set up the DataLoader
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=int(args.batch))
        
        elif 'ecthr' in args.tune:   
            # Set the maximum sequence length to 512 (maximum) since the ECtHR documents are very long
            max_seq_length = 512
            # Create DocDataset and RandomSampler objects to pass to the DataLoader
            # - Use the "DocDataset" class from "utils.py"
            train_dataset = DocDataset(args.tune, max_seq_length, tokenizer)
            train_sampler = RandomSampler(train_dataset)
            # Set up the DataLoader
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=int(args.batch))
            
            # Copy the MLM model for the final evaluation after training for sequence classification
            modelMLM = copy.deepcopy(model)
            
            # Re-Load the model as a SequenceClassification model 
            # - Since the weights are the same before fine-tuning, we can reload the model)
            if args.model == 'bert-base-uncased':
                model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.classes, output_attentions=False, output_hidden_states=False, return_dict=False)   
            else: 
                model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, output_attentions=False, output_hidden_states=False, return_dict=False) 
                              
        else:
            raise ValueError('Main: This code is arranged for GAP-Flipped and ECtHR-Gtuned. Modify the code for other datasets.')
                  
        # Set the parameters
        epochs = args.epoch
        learning_rate = args.lr
        epsilon = args.eps
        # Create a string for the tuning dataset name (for tuning and saving)
        tune_str = args.tune
        tune_str = tune_str.split('/')[-1]
        tune_str = tune_str.split('.')[0] 
        # Fine-tune the model
        model = fine_tune(model, tune_str, train_dataloader, epochs, learning_rate, epsilon, tokenizer, device)     

        # Create a string for the new debiased model name with fine-tuning dataset, learning rate, epsilon, and num of epochs           
        model_str = model_str + '-debiased_' + tune_str + '_lr' + str(learning_rate) + '_eps' + str(epsilon) + '_ep' + str(epochs)
        
        # Save the pretrained model and tokenizer to given directory
        model.bert.save_pretrained(args.out + 'models/' + model_str) 
        tokenizer.save_pretrained(args.out + 'models/' + model_str)
        
        # If the model is trained for ECtHR-GTuned, transfer the trained weights of the SequenceClassification model to the MLM copy for evaluation
        if 'ecthr' in args.tune:  
            modelMLM.load_state_dict(model.state_dict(), strict=False)
            model = modelMLM

        # Calculate post-association scores
        print('--- Calculating post-associations scores...')
        post_associations = model_evaluation(eval_data, tokenizer, model, device)
        print('--- Done.')
        
        # Add post-association scores to the dataframe
        eval_data = eval_data.assign(Post_Assoc=post_associations)

    else:
        print('--- No Fine-tuning today.')     
          
    # Save the results (association scores)
    out_path = args.out[:-1]
    eval_data.to_csv(args.out + 'results/' + model_str + '_' + eval_str + '.tsv', sep='\t', encoding='utf-8', index=False)
