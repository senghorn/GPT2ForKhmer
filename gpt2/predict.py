from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./final_tokenizer')
model = GPT2LMHeadModel.from_pretrained('./final_model')

# Text to be input to the model
input_text = 'សង្ក្រាន្តចូលមកនៅក្នុងថ្ងៃសៅរ៍ ៥កើត ខែ ចេត្រ ត្រូវនឹងថ្ងៃទី១៣ ខែមេសា ឆ្នាំ២០២៤ វេលាម៉ោង ២២:១៧ និង២៤ វិនាទីយប់ ។ ទេពធីតាមួយព្រះអង្គជាមគ្គនាយិការក្សាលោកនាឆ្នាំនេះ'
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
sample_outputs = model.generate(input_ids, do_sample=True, max_length=200, num_return_sequences=3)

# Open a text file to write output
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(f"Input: {input_text}\n")
    file.write("Generated Texts:\n")
    for i, sample_output in enumerate(sample_outputs):
        output_text = tokenizer.decode(sample_output, skip_special_tokens=True)
        file.write(f"{i}: {output_text}\n")

print("Output has been written to output.txt")
