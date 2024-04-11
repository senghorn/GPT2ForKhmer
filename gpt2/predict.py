from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('./final_tokenizer')
model = GPT2LMHeadModel.from_pretrained('./final_model')

# Generate text
input_ids = tokenizer.encode('ចាកចេញ ដោយ គ្មាន ការងារ', return_tensors='pt')
sample_outputs = model.generate(input_ids, do_sample=True, max_length=50, num_return_sequences=3)

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
