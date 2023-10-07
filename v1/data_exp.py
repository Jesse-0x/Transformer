from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import tokenizers
import torch
from torch.utils.data import DataLoader
import re

# Load the wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Preprocess the data
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    return sentences

# Apply preprocessing to the dataset
train_data = [clean_text(text) for text in dataset['train']['text']]
train_data = [sent for sublist in train_data for sent in sublist]  # Flatten the list of sentences

# Tokenize the dataset and pad the sentence to a fixed length
tokenizer = tokenizers.ByteLevelBPETokenizer()
tokenizer.train_from_iterator(train_data, vocab_size=1000, min_frequency=2, special_tokens=['<sos>', '<eos>', '<pad>'])
vocab = tokenizer.get_vocab()

# Tokenize and pad the sentences
train_data = ['<sos> ' + sent + ' <eos>' for sent in train_data]
train_data = [tokenizer.encode(sent).ids for sent in train_data]
max_len = max(len(sent) for sent in train_data)
train_data = [sent + [vocab['<pad>']] * (max_len - len(sent)) for sent in train_data]
train_data = torch.tensor(train_data)

# Create the dataloader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
