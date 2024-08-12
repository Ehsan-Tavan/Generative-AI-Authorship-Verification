"""
    Generative_AI_Authorship_Verification Project:
        data_loader:
            llm_trainer.py
"""
# ============================ Third Party libs =======================
import os
from datasets import Dataset

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.models import LLMModel
from src.data_preparation import sequence_classification_data_creator

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    TRAIN_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.single_train_file))
    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.single_dev_file))

    print(f"We have {len(TRAIN_DATA)} training samples.")
    print(f"We have {len(DEV_DATA)} validation samples.")

    TRAIN_SAMPLES, LABEL2ID, ID2LABEL = sequence_classification_data_creator(TRAIN_DATA)
    DEV_SAMPLES, _, _ = sequence_classification_data_creator(DEV_DATA)

    TRAIN_DATASET = Dataset.from_list(TRAIN_SAMPLES)
    DEV_DATASET = Dataset.from_list(DEV_SAMPLES)
    print(LABEL2ID)
    print(ID2LABEL)

    MODEL = LLMModel(model_path=ARGS.model_path, label2id=LABEL2ID, id2label=ID2LABEL, args=ARGS)
    MODEL.create_peft_model()

    MODEL.fine_tune(train_dataset=TRAIN_DATASET, eval_dataset=DEV_DATASET)
