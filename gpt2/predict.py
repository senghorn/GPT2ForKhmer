from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Paths where the model and tokenizer were saved
model_path = r"C:\Users\Sun_r\Projects\NLPProject\khmer-text-data\gpt2\fine_tuned_model"
tokenizer_path = r"C:\Users\Sun_r\Projects\NLPProject\khmer-text-data\gpt2\fine_tuned_tokenizer"

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Ensure that the model uses the same special tokens as the tokenizer
model.resize_token_embeddings(len(tokenizer))

# The text prompt to generate from
text_prompt = "ខ្ញុំចូលចិត្ត"

# Encode the prompt text
input_ids = tokenizer.encode(text_prompt, return_tensors='pt')

# Generate text
# Note: adjust the max_length as per your requirement
output = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:", generated_text)
