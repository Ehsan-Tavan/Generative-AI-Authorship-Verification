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
                                 default="train_data.jsonl")
        self.parser.add_argument("--dev_file", type=str,
                                 default="dev_file.jsonl")

    def get_config(self):
        """

        Returns:

        """
        self.add_path()
        return self.parser.parse_args()
