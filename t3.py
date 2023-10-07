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



import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import nltk

nltk.download('reuters')
from nltk.corpus import reuters

# Load and preprocess the data
# Here, we use the reuters corpus from nltk as an example. You can replace it with dialogue data.
corpus = [' '.join(reuters.words(fileid)) for fileid in reuters.fileids()]

# Tokenize the data using BPE tokenizer
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(corpus, trainer)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize input and target (you need to define how to construct input and target from your data)
        input_text = self.texts[idx]
        target_text = self.texts[idx]  # for simplicity, using the same text as target
        input_ids = tokenizer.encode(input_text).ids
        target_ids = tokenizer.encode(target_text).ids
        return torch.tensor(input_ids), torch.tensor(target_ids)


# Define hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Prepare DataLoader
train_dataset, val_dataset = train_test_split(corpus, test_size=0.1)
train_loader = DataLoader(TextDataset(train_dataset, tokenizer), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TextDataset(val_dataset, tokenizer), batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function


# hyperparameters
input_size = 1000
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

model = Transformer(num_layers=6, d_model=512, num_heads=8, d_ff=2048)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for input_ids, target_ids in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids, target_ids)
        loss = loss_function(outputs.squeeze(-1), target_ids)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            outputs = model(input_ids, target_ids)
            val_loss += loss_function(outputs.squeeze(-1), target_ids).item()
    print(f"Epoch {epoch}, Val Loss: {val_loss / len(val_loader)}")
