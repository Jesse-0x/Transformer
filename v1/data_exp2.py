from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Extract text data
text_data = dataset['train']['text']

# Split text into sentences
sentences = [sentence for text in text_data for sentence in text.split('\n') if sentence]

# Assume each even-indexed sentence is a question and each odd-indexed sentence is an answer
questions = sentences[::2]
answers = sentences[1::2]

# Prepare data for training and validation
train_questions, valid_questions, train_answers, valid_answers = train_test_split(questions, answers, test_size=0.1)

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Ensure that the tokenizer has a padding token set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize pairs of questions and answers
def tokenize_pair(question, answer):
    return tokenizer(question, answer, truncation=True, padding='max_length', max_length=512)

# Tokenize training and validation data
train_tokenized = [tokenize_pair(q, a) for q, a in zip(train_questions, train_answers)]
valid_tokenized = [tokenize_pair(q, a) for q, a in zip(valid_questions, valid_answers)]

# Set the batch size
batch_size = 32

# Prepare DataLoader
train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_tokenized, batch_size=batch_size)
