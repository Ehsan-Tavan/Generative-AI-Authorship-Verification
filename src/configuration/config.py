"""
    Generative_AI_Authorship_Verification Project:
        configuration:
                config.py
"""

# ============================ Third Party libs ============================
import argparse
from pathlib import Path


class BaseConfig:
    """
        BaseConfig:
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str,
                                 default="Paraphraser_Bart")
        self.parser.add_argument("--load_in_8bit",
                                 type=bool,
                                 default=False)
        self.parser.add_argument("--load_in_4bit",
                                 type=bool,
                                 default=True)
        self.parser.add_argument("--bnb_4bit_use_double_quant",
                                 type=bool,
                                 default=True)
        self.parser.add_argument("--lora_alpha",
                                 type=int,
                                 default=16)
        self.parser.add_argument("--lora_dropout",
                                 type=int,
                                 default=0.05)
        self.parser.add_argument("--lora_rank",
                                 type=int,
                                 default=64)
        self.parser.add_argument("--device", default="cuda:0",
                                 help="device to inference models on it")
        self.parser.add_argument("--per_device_train_batch_size",
                                 type=int,
                                 default=8)
        self.parser.add_argument("--gradient_accumulation_steps",
                                 type=int,
                                 default=4)
        self.parser.add_argument("--optim",
                                 type=str,
                                 help="activates the paging for better memory management",
                                 default="paged_adamw_32bit")
        self.parser.add_argument("--num_train_epochs",
                                 type=int,
                                 default=10)
        self.parser.add_argument("--min_epochs",
                                 type=int,
                                 default=5)

        self.parser.add_argument("--evaluation_strategy",
                                 type=str,
                                 default="steps")
        self.parser.add_argument("--save_strategy",
                                 type=str,
                                 help="checkpoint save strategy to adopt during training [epoch]",
                                 default="steps")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate for AdamW optimizer",
                                 default=2e-5)
        self.parser.add_argument("--max_grad_norm",
                                 type=float,
                                 help="maximum gradient norm (for gradient clipping)",
                                 default=0.3)
        self.parser.add_argument("--warmup_ratio",
                                 type=float,
                                 help="number of steps used for a linear warmup from 0 "
                                      "to learning_rate",
                                 default=0.03)
        self.parser.add_argument("--lr_scheduler_type",
                                 type=str,
                                 default="constant")  # constant, cosine

        self.parser.add_argument("--stride",
                                 type=int,
                                 default=512)
        self.parser.add_argument("--max_length",
                                 type=int,
                                 default=800)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 default=4)
        self.parser.add_argument("--save_top_k",
                                 type=int,
                                 default=2)
        self.parser.add_argument("--training_data_type",
                                 type=str,
                                 default="text_pair")

        self.parser.add_argument("--binoculars_accuracy_threshold",
                                 type=float,
                                 default=0.9015310749276843)
        self.parser.add_argument("--binoculars_fpr_threshold",
                                 type=float,
                                 default=0.8536432310785527)
        self.parser.add_argument("--openai_api_key", type=str,
                                 default="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")

    def add_path(self) -> None:
        """
        function to add file path

        Returns:
            None

        """
        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Raw/")
        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Processed/")

        self.parser.add_argument("--pair_train_file", type=str,
                                 default="pair_train_data.json")
        self.parser.add_argument("--pair_dev_file", type=str,
                                 default="pair_dev_data.json")

        self.parser.add_argument("--single_train_file", type=str,
                                 default="single_train_data.json")
        self.parser.add_argument("--single_dev_file", type=str,
                                 default="single_dev_data.json")

        self.parser.add_argument("--saved_model_path",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() +
                                         "/assets/saved_model")
        self.parser.add_argument("--llama_model_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/llama-2-7b")
        self.parser.add_argument("--llama_peft_model_path",
                                 type=str,
                                 default="/mnt/disk2/ehsan.tavan/gen_ai/assets/saved_model/"
                                         "Generative_AI_Authorship_Verification_LLama/"
                                         "version_0/checkpoint-868")
        self.parser.add_argument("--mistral_model_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/Mistral-7B-v0.1")
        self.parser.add_argument("--mistral_peft_model_path",
                                 type=str,
                                 default="/mnt/disk2/ehsan.tavan/gen_ai/assets/saved_model/"
                                         "Generative_AI_Authorship_Verification_mistral/version_1/"
                                         "checkpoint-868")
        self.parser.add_argument("--lm_model_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/xlm-roberta-base")

        self.parser.add_argument("--observer_name_or_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/falcon-7b")
        self.parser.add_argument("--performer_name_or_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/falcon-7b-instruct")
        self.parser.add_argument("--outputDir",
                                 type=str,
                                 default="./output_file.jsonl")
        self.parser.add_argument("--inputDataset",
                                 type=str,
                                 default="./input_file.jsonl")
        self.parser.add_argument("--bart_model_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/bart_large")


    def get_config(self):
        """

        Returns:

        """
        self.add_path()
        return self.parser.parse_args()
