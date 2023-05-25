# fine_tune.py:  The fine-tuning process for GAP-Flipped and ECtHR-Gtuned datasets
#                Can be modified for use with other datasets
#
# Author:        Mustafa Bozdag
# Date:          04/28/2023

import time
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import format_time, mask_tokens

# Fine-tuning function:
# - model: Bert/AutoModel for MLM/Sequence Classification
# - dataset: GAP-Flipped/ECtHR-Gtuned
# - dataloader: Corresponding dataloader for the dataset
# - epochs: Number of epochs
# - learning_rate: Learning rate
# - epsilon: Epsilon value for AdamW optimizer
# - tokenizer: Bert/AutoModel tokenizer
# - device: GPU/CPU
# - Returns the debiased model
def fine_tune(model, dataset, dataloader, epochs, learning_rate, epsilon, tokenizer, device):

    # Put the model on the device and train mode
    model.to(device)
    model.train()

    # This part is mainly from the tutorial:
    # https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=oCYZa1lQ8Jn8&forceEdit=true&sandboxMode=true
    # The AdamW optimizer below is from Transformers (Adam with weight-decay fix), NOT from Torch 
    optimizer = AdamW(model.parameters(), 
                        lr=learning_rate, # default: 5e-5
                        eps=epsilon)      # default: 1e-8   

    # Total number of training steps = (num of batches)*(num of epochs)
    total_steps = len(dataloader) * epochs
    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Start fine-tuning
    print('\n--- Fine-tuning:')
    for epoch_i in range(0, epochs):
        print('\n=== Epoch {:} / {:} ==='.format(epoch_i + 1, epochs))

        # Measure how long the epoch takes
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0

        
        for step, batch in enumerate(dataloader):
            # Print the progress every 100 batches
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes and report progress
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}  -  Elapsed: {:}'.format(step, len(dataloader), elapsed))
            
            if dataset=='gap-flipped':
                # Mask the inputs for GAP-Flipped so the model can actually learn something
                b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
                b_input_ids = b_input_ids.to(device)
                b_labels = b_labels.to(device)
                b_input_mask = batch[1].to(device)
                              
            elif dataset=='ecthr-gtuned':
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)
            
            else:
                raise ValueError('fine-tune.py: This code is arranged for GAP-Flipped and ECtHR-Gtuned. Modify the code for other datasets.')
            
            # Set gradients to zero before backpropagation
            model.zero_grad()
            # Forward propagation, returns outputs as a tuple in the form [loss, logits]
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]            
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
            
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(dataloader)


        print('')
        print('  Average training loss: {0:.2f}'.format(avg_train_loss))
        print('  Training epoch took: {:}'.format(format_time(time.time() - t0)))

    print('\nFine-tuning done.')

    return model