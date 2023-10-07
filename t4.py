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
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Ensure that the tokenizer has a padding token set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set the batch size
batch_size = 32

# Data collator is used to pad the sequences to the same length
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Prepare DataLoader
train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
valid_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)




# Hyperparameters
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1
learning_rate = 0.0001
vocab_size = len(tokenizer.get_vocab())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the Transformer model
model = Transformer(vocab_size, num_layers, d_model, num_heads, d_ff, dropout)
model.to(device)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the loss function
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        # Extract input and target from batch
        input_ids = batch['input_ids'].to(device)
        target_ids = input_ids.clone()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, target_ids)

        # Shift the outputs and the target ids to the left to exclude the last token
        shifted_outputs = outputs[:, :-1, :].contiguous()
        shifted_target_ids = target_ids[:, 1:].contiguous()

        # Compute loss and perform a step of optimization
        loss = criterion(shifted_outputs.view(-1, shifted_outputs.size(-1)), shifted_target_ids.view(-1))
        # print oout the output from model to see what it looks like, convert to text, also the expected output
        print(tokenizer.decode(torch.argmax(outputs[0], dim=1).tolist()))
        print(tokenizer.decode(torch.argmax(shifted_outputs[0], dim=1).tolist()))
        print(tokenizer.decode(shifted_target_ids[0].tolist()))
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

    # Validation loop
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = input_ids.clone()
            outputs = model(input_ids, target_ids)
            shifted_outputs = outputs[:, :-1, :].contiguous()
            shifted_target_ids = target_ids[:, 1:].contiguous()
            loss = criterion(shifted_outputs.view(-1, shifted_outputs.size(-1)), shifted_target_ids.view(-1))
            total_loss += loss.item()

    print(f'Epoch: {epoch}, Val Loss: {total_loss / len(valid_loader)}')

# Save the model
torch.save(model.state_dict(), 'transformer.pt')