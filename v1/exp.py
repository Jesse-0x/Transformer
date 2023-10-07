import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.tokenize import word_tokenize
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling

# from data import train_loader, valid_loader, tokenizer
from data_exp import train_loader, tokenizer, vocab
from model import Transformer

# Hyperparameters
# Define the model
# Hyperparameters
vocab_size = len(tokenizer.get_vocab())
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1
padding_id = 2  # Updated
max_grad_norm = 1.0  # For gradient clipping

# Define the model
model = Transformer(vocab_size, num_layers, d_model, num_heads, d_ff, dropout)

# Define Loss, Optimizer, and Learning Rate Scheduler
criterion = nn.CrossEntropyLoss(ignore_index=padding_id)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / ((epoch + 1) ** 0.5))
device = torch.device('mps')
# Training Loop
import numpy as np


# Usage:
# Assume padding_id = 0
# src, tgt are your source and target tensors with shapes (batch_size, src_len) and (batch_size, tgt_len) respectively.
# tgt_mask = create_tgt_mask(tgt, padding_id=0, num_heads=32)

def create_tgt_mask(tgt, padding_id, num_heads):
    tgt_len = tgt.size(1)
    # Create target padding mask
    tgt_pad_mask = (tgt != padding_id).unsqueeze(-2)  # Shape: (batch_size, 1, tgt_len)

    # Create target subsequent mask
    tgt_sub_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()  # Shape: (tgt_len, tgt_len)

    # Combine the two masks
    tgt_mask = tgt_pad_mask & tgt_sub_mask  # Shape: (batch_size, tgt_len, tgt_len)

    # Reshape the mask to get the desired shape (N * num_heads, tgt_len, tgt_len)
    batch_size = tgt.size(0)
    tgt_mask_expanded = tgt_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).contiguous().view(batch_size * num_heads, tgt_len, tgt_len)  # Shape: (batch_size * num_heads, tgt_len, tgt_len)

    return tgt_mask_expanded

def create_src_mask(src, pad_id):
    src_mask = (src != pad_id).unsqueeze(-2).unsqueeze(1)  # Shape: (batch_size, 1, 1, src_len)
    return src_mask




# load the model
model.load_state_dict(torch.load('/Users/jesse/PycharmProjects/Transformer/working.pth'))
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids = batch
        target_ids = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        # create src and target masks

        # Forward pass
        outputs = model(input_ids, target_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))

        # print out the model's prediction
        print(torch.argmax(outputs[0], dim=1).tolist())
        print(tokenizer.decode(torch.argmax(outputs[0], dim=1).tolist()))
        print(tokenizer.decode(target_ids[0].tolist()))
        # Check for NaNs
        if torch.isnan(loss).any():
            raise ValueError("Loss is NaN")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update parameters
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Iteration {i + 1}, Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

userinput = str(input("pls input a word"))
print(userinput[0] + userinput[1:].replace(userinput[0], "?"))

