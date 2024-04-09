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
                                 default="Generative_AI_Authorship_Verification")
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
                                 default=0.1)
        self.parser.add_argument("--lora_rank",
                                 type=int,
                                 default=64)
        self.parser.add_argument("--device", default="cuda:0",
                                 help="device to inference models on it")
        self.parser.add_argument("--per_device_train_batch_size",
                                 type=int,
                                 default=32)
        self.parser.add_argument("--gradient_accumulation_steps",
                                 type=int,
                                 default=1)
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
                                 default=512)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 default=4)

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
        self.parser.add_argument("--train_file", type=str,
                                 default="train_single.csv")
        self.parser.add_argument("--dev_file", type=str,
                                 default="dev_single.csv")
        self.parser.add_argument("--saved_model_path",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() +
                                         "/assets/saved_model")
        self.parser.add_argument("--model_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/xlm-roberta-base")

    def get_config(self):
        """

        Returns:

        """
        self.add_path()
        return self.parser.parse_args()
