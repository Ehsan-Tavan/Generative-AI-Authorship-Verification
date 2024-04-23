"""
    Generative_AI_Authorship_Verification Project:
        data_loader:
            paraphrase_inferencer.py
"""
# ============================ Third Party libs =======================

import os

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.data_preparation import paraphraser_data_creator


def remove_text_a(text_a, text_b):
    return text_b.replace(text_a, '')


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))

    PROCESSED_DEV_DATA = paraphraser_data_creator(data=DEV_DATA, mode="test")

    DEV_DATASET = Dataset.from_list(PROCESSED_DEV_DATA)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_path,  # Llama 2 7B, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True
    )

    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_path, add_bos_token=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_model = PeftModel.from_pretrained(base_model, "/mnt/disk2/ehsan.tavan/gen_ai/assets/"
                                                       "saved_model/Paraphraser/version_0/"
                                                       "checkpoint-1080")
    peft_model.eval()

    for index, sample in enumerate(DEV_DATASET):
        print(index / len(DEV_DATA) * 100)
        tokenized_text = tokenizer(sample["instruction"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = peft_model.generate(**tokenized_text, max_new_tokens=500,
                                         pad_token_id=tokenizer.eos_token_id)

        results = [tokenizer.decode(res, skip_special_tokens=True) for res in output]

        generated_text = remove_text_a(text_a=sample["instruction"], text_b=results[0])

        print(generated_text)
        print()
        print("##################")
