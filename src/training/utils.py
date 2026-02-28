from transformers import DataCollatorForLanguageModeling


def build_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
