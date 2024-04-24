"""
    Generative_AI_Authorship_Verification Project:
        data_loader:
            paraphraser_trainer.py
"""
# ============================ Third Party libs =======================
import os

from datasets import Dataset
from transformers import AutoTokenizer
# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.models.paraphraser_model import LLMModel
from src.data_preparation import paraphraser_data_creator

os.environ["WANDB_MODE"] = "offline"

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TRAIN_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_train_file))
    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))

    print(f"We have {len(TRAIN_DATA)} training samples.")
    print(f"We have {len(DEV_DATA)} validation samples.")

    TRAIN_SAMPLES = paraphraser_data_creator(TRAIN_DATA)
    DEV_SAMPLES = paraphraser_data_creator(DEV_DATA)

    TOKENIZER = AutoTokenizer.from_pretrained(ARGS.model_path, trust_remote_code=True)

    TRAIN_DATASET = Dataset.from_list(TRAIN_SAMPLES)
    DEV_DATASET = Dataset.from_list(DEV_SAMPLES)

    MODEL = LLMModel(model_path=ARGS.model_path, args=ARGS)
    MODEL.create_peft_model()

    MODEL.fine_tune(train_dataset=TRAIN_DATASET, eval_dataset=DEV_DATASET)
