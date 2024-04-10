from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode a text inputs
text = "The only thing that really matters in life is"
indexed_tokens = tokenizer.encode(text, add_special_tokens=True)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# TODO: We have to try using CUDA compatible machine
# tokens_tensor = tokens_tensor.to('cuda')
# model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]


predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode([predicted_index])

print(predicted_text)
