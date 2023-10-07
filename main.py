import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.fc = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear layers
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Final linear layer
        output = self.fc(output)

        return output


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked multi-head attention (for target sequence)
        attn_output1 = self.mha1(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output1))

        # Multi-head attention (using encoder's output)
        attn_output2 = self.mha2(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output2))

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers, input_vocab_size,
                 target_vocab_size, dropout=0.1):
        super(Transformer, self).__init__()

        # Embeddings
        self.src_emb = torch.nn.Embedding(input_vocab_size, d_model)
        self.tgt_emb = torch.nn.Embedding(target_vocab_size, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # Encoder and Decoder stacks
        self.encoders = torch.nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoders = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])

        # Output layer
        self.fc = torch.nn.Linear(d_model, target_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_enc(self.src_emb(src))
        tgt = self.pos_enc(self.tgt_emb(tgt))

        # Encoder stack
        for encoder in self.encoders:
            src = encoder(src, src_mask)

        # Decoder stack
        for decoder in self.decoders:
            tgt = decoder(tgt, src, src_mask, tgt_mask)

        # Final linear layer
        output = self.fc(tgt)

        return output



import torch.optim as optim
import tokenizers
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import nltk
from nltk.corpus import brown

# Dataset
# We will use the Brown corpus from NLTK
class BrownDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data
        self.vocab = self.tokenizer.get_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sentence
        sentence = self.data[idx]
        # Tokenize the sentence
        sentence = self.tokenizer.encode(sentence)
        # Convert the tokens to ids
        sentence = sentence.ids
        # Convert the ids to tensors
        sentence = torch.tensor(sentence)
        # Return the sentence
        return sentence[:-1], sentence[1:]

    def get_vocab(self):
        return self.vocab


# Tokenizer based on BPE
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = Sequence([NFKC()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

special_tokens = ["<srt>","<pad>","<unk>","<mask>"]
trainer = BpeTrainer(vocab_size=5000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=special_tokens)

# load the Brown corpus from the brown1.txt file
data = []
passage = ""
with open('data/brown1.txt', 'r') as f:
    for line in f:
        c = line.split(' ')
        if passage != c[0]:
            passage = c[0]
            data.append(line[9:][:-1])
        else:
            data[-1] += line[9:][:-1]

# add <srt> and <pad> tokens
data = ["<srt> " + sentence + " <pad>" for sentence in data]

# Get the data, for each sentence, we add a <s> at the beginning and a <pad> at the end

tokenizer.train_from_iterator(data, trainer=trainer)

# Save
tokenizer.save("tokenizer.json")

# Load
tokenizer = Tokenizer.from_file("tokenizer.json")


def pad_sequence(sequences, batch_first=False, padding_value=0):
    # Find the max length of sequences in the batch
    max_length = max([len(seq) for seq in sequences])

    # Initialize a tensor of size [batch_size, max_length] filled with the padding value
    if batch_first:
        padded_sequences = torch.full((len(sequences), max_length), padding_value, dtype=torch.int64)
    else:
        padded_sequences = torch.full((max_length, len(sequences)), padding_value, dtype=torch.int64)

    # Copy values from sequences to the appropriate positions in the padded_sequences tensor
    for i, seq in enumerate(sequences):
        if batch_first:
            padded_sequences[i, :len(seq)] = seq
        else:
            padded_sequences[:len(seq), i] = seq

    return padded_sequences

def collate_fn(batch):
    # Split the batch into source and target sequences
    src_sequences, tgt_sequences = zip(*batch)

    # Pad the sequences
    src_sequences_padded = pad_sequence(src_sequences, batch_first=True)
    tgt_sequences_padded = pad_sequence(tgt_sequences, batch_first=True)

    return src_sequences_padded, tgt_sequences_padded


train_dataset = BrownDataset(tokenizer, data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Hyperparameters
d_model = 256
num_heads = 8
d_ff = 512
num_encoder_layers = 3
num_decoder_layers = 3
input_vocab_size = len(tokenizer.get_vocab())
target_vocab_size = len(tokenizer.get_vocab())
dropout = 0.1

# Model, Loss, Optimizer
model = Transformer(d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers, input_vocab_size,
                    target_vocab_size, dropout)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10
print("Start training...")
model.train()
for epoch in range(epochs):
    total_loss = 0.0

    for src, tgt in train_loader:
        # Shift target sequence for teacher forcing
        tgt_input = tgt[:, :-1]
        ground_truth = tgt[:, 1:]

        optimizer.zero_grad()

        outputs = model(src, tgt_input)
        outputs = outputs.view(-1, outputs.size(-1))
        ground_truth = ground_truth.view(-1)

        loss = criterion(outputs, ground_truth)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{train_loader}], Loss: {avg_loss:.4f}")



# eval
model.eval()
src = train_dataset[0][0].unsqueeze(0).to('cpu')
tgt_input = train_dataset[0][1][0].unsqueeze(0).to('cpu')
output = model(src, tgt_input)
output = output.argmax(dim=-1).squeeze(0)
print(" ".join([train_dataset.vocab.get_itos()[idx] for idx in output]))


