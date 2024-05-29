# data_utils.py
import re
import string

# data_utils.py
def load_dataset(file_path):
    """Load the dialog dataset from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        dialog_data = file.readlines()
    return dialog_data

def preprocess_data(dialog_data):
    """Preprocess the dialog data."""
    processed_data = []
    for line in dialog_data:
        # Remove leading/trailing whitespace and newline characters
        line = line.strip()
        # Tokenize the dialog
        tokens = tokenize(line)
        # Filter out empty lines
        if tokens:
            processed_data.append(tokens)
    return processed_data

def tokenize(text):
    """Tokenize a line of text."""
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize by splitting on whitespace
    tokens = text.split()
    return tokens

def preprocess_input(user_input):
    """Preprocess user input before feeding it to the model."""
    # Tokenize the input
    tokens = tokenize(user_input)
    # Convert tokens to tensors, encode with appropriate vocabulary
    # This step depends on the specific requirements of your model
    processed_input = encode_tokens(tokens)
    return processed_input

def encode_tokens(tokens):
    """Encode tokens with appropriate vocabulary."""
    # Placeholder function
    # Depending on your model architecture, you may need to implement token encoding differently
    # For example, you might use word embeddings or one-hot encoding
    # This function should convert tokens to tensors that can be input to your model
    return tokens