"""
    Generative_AI_Authorship_Verification Project:
        data_loader:
            instruction_inferencer.py
"""
# ============================ Third Party libs =======================
import os
import re

import torch
from datasets import Dataset

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.data_preparation import instruction_tuning_data_creator
from src.models.instruction_tuning_model import LLMModel


def remove_text_a(text_a, text_b):
    return text_b.replace(text_a, '')


def extract_answer(answer: str):
    # response_pattern = re.compile(r'### Response:(.*?)### End', re.DOTALL)
    # response_pattern = re.compile(r'### Response:\n(.+)\n', re.DOTALL)

    response_pattern = re.compile(r'### Response:(.*?)### End(.*?)', re.DOTALL)

    match = response_pattern.search(answer)
    if match:
        extracted_response = match.group(1).strip()
        return extracted_response
    else:
        print(answer)
        return answer


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))
    print(f"We have {len(DEV_DATA)} validation samples.")

    DEV_SAMPLES = instruction_tuning_data_creator(DEV_DATA, mode="test")
    DEV_DATASET = Dataset.from_list(DEV_SAMPLES)

    MODEL = LLMModel(model_path=ARGS.llama_model_path, args=ARGS)
    PEFT_MODEL = MODEL.load_peft_model(peft_model_path="/mnt/disk2/ehsan.tavan/gen_ai/assets/"
                                                       "saved_model/Instruction_Tuning/"
                                                       "version_0/checkpoint-130")

    for index, sample in enumerate(DEV_DATASET):
        print(index / len(DEV_DATASET) * 100)
        tokenized_text = MODEL.tokenizer(sample["instruction"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = PEFT_MODEL.generate(**tokenized_text, max_new_tokens=20,
                                         pad_token_id=MODEL.tokenizer.eos_token_id)

            results = [MODEL.tokenizer.decode(res, skip_special_tokens=True) for res in output]

            generated_text = remove_text_a(text_a=sample["instruction"], text_b=results[0])

            generated_text = "### Response:" + generated_text

            extracted_answer = extract_answer(generated_text)  # .strip().replace("'", "")
            print(extracted_answer)
