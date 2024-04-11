from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer_path = r"khmer_tokenizer" 

tokenizer = GPT2Tokenizer(vocab_file=f"{tokenizer_path}\\vocab.json", merges_file=f"{tokenizer_path}\\merges.txt")

# Adding the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Indicate the special tokens to the model
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})
model.config.pad_token_id = model.config.eos_token_id

train_file_path = "../data/seg_kmwiki_data/kmwiki_data/ប្រវត្តិសាស្ត្រខ្មែរ.txt" 
valid_file_path = "../data/seg_kmwiki_data/kmwiki_data/ការលួចចម្លង.txt"

train_dataset = TextDataset(
  tokenizer=tokenizer,
  file_path=train_file_path,
  block_size=128)

valid_dataset = TextDataset(
  tokenizer=tokenizer,
  file_path=valid_file_path,
  block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-khmer",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=500,
    warmup_steps=500,
    prediction_loss_only=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Start fine-tuning
# trainer.train()

save_model_path = "fine_tuned_model"
save_tokenizer_path = "fine_tuned_tokenizer"

# Save the model and tokenizer
model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_tokenizer_path)