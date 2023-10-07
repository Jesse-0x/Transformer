import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split


# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.mha(x, x, x, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        return out2


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.mha2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        print("X shape:", x.shape)
        attn1_output, attn1_weights = self.mha1(x, x, x, attn_mask=tgt_mask)
        attn1_output = self.dropout1(attn1_output)
        out1 = self.norm1(x + attn1_output)
        if len(out1.shape) == 2:
            out1 = out1.unsqueeze(1)
        attn2_output, attn2_weights = self.mha2(out1, enc_output, enc_output, attn_mask=src_mask)
        attn2_output = self.dropout2(attn2_output)
        out2 = self.norm2(out1 + attn2_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.norm3(out2 + ffn_output)
        return out3


class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        print("Source shape:", src.shape)
        print("Target shape:", tgt.shape)
        src = self.pos_encoder(src)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.linear(dec_output)
        return output

# hyperparameters
input_size = 1000
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

# model
model = Transformer(num_layers, d_model, num_heads, d_ff, dropout)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# loss function
criterion = nn.CrossEntropyLoss()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use nltk as training dataset and tokenizer to tokenize the text
import nltk
import tokenizers
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg

# load the dataset
nltk.download('gutenberg')
nltk.download('punkt')
corpus = gutenberg.sents('bible-kjv.txt')[:1000]
corpus = [' '.join(sent) for sent in corpus]

# split the dataset into train and test
train, test = train_test_split(corpus, test_size=0.2)

# create the vocabulary
tokenizer = tokenizers.ByteLevelBPETokenizer()
tokenizer.train_from_iterator(train, vocab_size=1000, min_frequency=2, special_tokens=['<sos>', '<eos>', '<pad>'])

# create the vocabulary
vocab = tokenizer.get_vocab()

# tokenize the dataset and pad the sentence to 512 tokens
train = ['<sos> ' + sent + ' <eos>' for sent in train]
train = [tokenizer.encode(sent).ids for sent in train]
train = [sent + [vocab['<pad>']] * (512 - len(sent) + 1) for sent in train]
train = torch.tensor(train)

test = ['<sos> ' + sent + ' <eos>' for sent in test]
test = [tokenizer.encode(sent).ids for sent in test]
test = [sent + [vocab['<pad>']] * (512 - len(sent) + 1) for sent in test]
test = torch.tensor(test)


# create the dataloader
train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)


# train the model
model.train()
model.to(device)
for epoch in range(10):
    for i, batch in enumerate(train_loader):
        src = batch[:, :-1].float().to(device)
        tgt = batch[:, 1:].float().to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# test the model
model.eval()
model.to(device)
for i, batch in enumerate(test_loader):
    src = batch[:, :-1].to(device)
    tgt = batch[:, 1:].to(device)
    output = model(src, tgt)
    loss = criterion(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
    print(f'Loss: {loss.item()}')
    break

# save the model
torch.save(model.state_dict(), 'transformer.pt')

# load the model
model.load_state_dict(torch.load('transformer.pt'))

# predict with input
model.eval()
model.to(device)
text = 'I love'
tokens = word_tokenize(text)
tokens = ['<sos>'] + tokens + ['<eos>']
tokens = [vocab.stoi[token] for token in tokens]
src = torch.tensor(tokens).unsqueeze(0).to(device)
tgt = torch.tensor([vocab.stoi['<sos>']]).unsqueeze(0).to(device)
for i in range(10):
    output = model(src, tgt)
    tgt = torch.cat((tgt, output.argmax(-1)[:,-1].unsqueeze(1)), dim=1)
    print(vocab.itos[tgt[0, -1].item()])
