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
    # remove the '\n' from the end of each sentence
    text = re.sub(r'\n', '', text)
    # also remove the initial spaces and the spaces at the end
    text = text.strip()
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

# everything excpet the last word is the source, and everything except the first word is the target
# remember to remove word not just letter
def remove_word(sent):
    # remove the last 1/3 of the sentence
    rm = sent.split()[:-len(sent.split())//3]
    return ' '.join(rm)

def remove_first_word(sent):
    return ' '.join(sent.split()[2:])

src_data = [remove_word(sent) for sent in train_data]
tgt_data = [remove_first_word(sent) for sent in train_data]

# find all the empty sentences in src_data and then remove the corresponding sentences in tgt_data
empty_idx = [i for i, sent in enumerate(src_data) if len(sent) == 0]
src_data = [sent for i, sent in enumerate(src_data) if i not in empty_idx]
tgt_data = [sent for i, sent in enumerate(tgt_data) if i not in empty_idx]



# Tokenize and pad the sentences
# train_data = ['<sos> ' + sent + ' <eos>' for sent in train_data]
# train_data = [tokenizer.encode(sent).ids for sent in train_data]
# max_len = max(len(sent) for sent in train_data)
# train_data = [sent + [vocab['<pad>']] * (max_len - len(sent)) for sent in train_data]
# train_data = torch.tensor(train_data)


src_data = ['<sos> ' + sent + ' <eos>' for sent in src_data]
src_data = [tokenizer.encode(sent).ids for sent in src_data]

tgt_data = ['<sos> ' + sent + ' <eos>' for sent in tgt_data]
tgt_data = [tokenizer.encode(sent).ids for sent in tgt_data]

max_len = max(max(len(sent) for sent in src_data), max(len(sent) for sent in tgt_data))

src_data = [sent + [vocab['<pad>']] * (max_len - len(sent)) for sent in src_data]
src_data = torch.tensor(src_data)


tgt_data = [sent + [vocab['<pad>']] * (max_len - len(sent)) for sent in tgt_data]
tgt_data = torch.tensor(tgt_data)

# Create the dataloader

# Create a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return {
            'src': self.src_data[idx],
            'tgt': self.tgt_data[idx]
        }

# Create instances of the custom dataset
train_dataset = CustomDataset(src_data, tgt_data)


# Create the dataloader
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
