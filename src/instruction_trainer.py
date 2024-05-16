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


os.environ["WANDB_MODE"] = "offline"
# set the wandb project where this run will be logged

# save your trained model checkpoint to wandb
# os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
# os.environ["WANDB_WATCH"] = "false"
# os.environ["WANDB_API_KEY"] = "local-ae409c7d636efeee746f95149fa44fbefb2ae93e"

seed_value = 42
random.seed(seed_value)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    model_name = ["SOLAR-10.7B-Instruct_v1.0", "Mistral-7B-v0.1", "llama-2-13b",
                  "SOLAR-10.7B-v1.0"]
    model_path = [
                  "/mnt/disk2/LanguageModels/SOLAR-10.7B-Instruct-v1.0",
                  "/mnt/disk2/LanguageModels/Mistral-7B-v0.1",
                  "/mnt/disk2/LanguageModels/llama-2-13b",
                  "/mnt/disk2/LanguageModels/SOLAR-10.7B-v1.0"]

    TRAIN_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_train_file))
    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))
    print(f"We have {len(TRAIN_DATA)} training samples.")
    print(f"We have {len(DEV_DATA)} validation samples.")

    TRAIN_SAMPLES = instruction_tuning_data_creator(TRAIN_DATA)
    TRAIN_DATASET = Dataset.from_list(TRAIN_SAMPLES)

    DEV_SAMPLES = instruction_tuning_data_creator(DEV_DATA)
    DEV_DATASET = Dataset.from_list(DEV_SAMPLES)

    # print(wandb.__version__)
    # wandb.init(project="my_wandb_test_project") local-112dc40780d62200a2b2ac380f03565c766bbdd9

    # for i in range(len(model_name)):
    #     ARGS.model_path = model_path[i]
    #     ARGS.model_name = model_name[i]
    MODEL = LLMModel(model_path=ARGS.llama_model_path, args=ARGS)
    MODEL.create_peft_model()
    MODEL.fine_tune(train_dataset=TRAIN_DATASET, eval_dataset=DEV_DATASET)
    # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()
