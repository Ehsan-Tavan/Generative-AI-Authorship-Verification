"""
    Generative_AI_Authorship_Verification Project:
        data_loader:
            instruction_trainer.py
"""
# ============================ Third Party libs =======================
import os
import random
from datasets import Dataset

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_preparation import instruction_tuning_data_creator
from src.models.instruction_tuning_model import LLMModel
from src.data_loader import load_jsonl


seed_value = 42
random.seed(seed_value)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TRAIN_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_train_file))
    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))
    print(f"We have {len(TRAIN_DATA)} training samples.")
    print(f"We have {len(DEV_DATA)} validation samples.")

    TRAIN_SAMPLES = instruction_tuning_data_creator(TRAIN_DATA)
    TRAIN_DATASET = Dataset.from_list(TRAIN_SAMPLES)

    DEV_SAMPLES = instruction_tuning_data_creator(DEV_DATA)
    DEV_DATASET = Dataset.from_list(DEV_SAMPLES)

    MODEL = LLMModel(model_path=ARGS.llama_model_path, args=ARGS)
    MODEL.create_peft_model()
    MODEL.fine_tune(train_dataset=TRAIN_DATASET, eval_dataset=DEV_DATASET)
