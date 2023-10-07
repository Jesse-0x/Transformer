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
