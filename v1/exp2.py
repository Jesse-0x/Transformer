import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm

from data_exp import train_loader, tokenizer, vocab
from model import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the Transformer model
vocab_size = len(tokenizer.get_vocab())
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

model = Transformer(vocab_size, num_layers, d_model, num_heads, d_ff, dropout)
model.to(device)  # Assuming CUDA is available

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=2)  # Assuming pad_token_id is available from tokenizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


def create_src_mask(src, pad_id=2):
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    return src_mask


def create_tgt_mask(tgt, padding_id=2, num_heads=32):
    tgt_mask = (tgt != padding_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_mask & torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=device)).bool()
    tgt_mask = tgt_mask.repeat(num_heads, 1, 1)
    tgt_mask = tgt_mask.float()
    return tgt_mask


# load the model
model.load_state_dict(torch.load('/Users/jesse/PycharmProjects/Transformer/v1/v1.3.pth'))
# Training Loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        # Move data to GPU
        src = batch['src']
        tgt = batch['tgt']

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(src, tgt, create_src_mask(src), create_tgt_mask(tgt))
        # The dimension of outputs is [batch_size, seq_len, vocab_size], and the dimension of tgt is [batch_size, seq_len]
        # So, we need to reshape the tensors to compute loss
        loss = criterion(outputs.view(-1, vocab_size), tgt.view(-1))
        # print out the model's prediction and the expected output
        print('model prediction: ', end='')
        print(tokenizer.decode(torch.argmax(outputs[0], dim=1).tolist()))
        print('expected output:  ', end='')
        print(tokenizer.decode(tgt[0].tolist()))
        print('model input:      ', end='')
        print(tokenizer.decode(src[0].tolist()))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')

    # Print loss for this epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

# Save the model
torch.save(model.state_dict(), 'v1.4.pth')
