from tokenizers import ByteLevelBPETokenizer
import os

# Ensure the model directory exists before starting the tokenizer training
model_dir = f'C:\Users\Sun_r\OneDrive\Documents\KhmerNLP\gpt2\khmer_tokenizer'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training with your Khmer dataset
# Ensure the dataset path is correctly formatted for Windows, or consider using a raw string or double backslashes
dataset_path = r'C:\Users\Sun_r\OneDrive\Documents\KhmerNLP\gpt2\data-set.txt'

# Tokenizer trains on the dataset with UTF-8 encoding by default, suitable for Khmer text
tokenizer.train(files=dataset_path, vocab_size=52000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save the tokenizer model to the prepared directory
tokenizer.save_model(model_dir)
