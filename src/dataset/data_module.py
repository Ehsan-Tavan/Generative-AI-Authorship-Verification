"""
    Generative_AI_Authorship_Verification Project:
        dataset:
            data_module.py
"""

# ============================ Third Party libs ============================
import argparse
import pytorch_lightning as pl
import torch
import transformers


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: list,
                 test_data: list,
                 dev_data: list,
                 config: argparse.ArgumentParser.parse_args,
                 dataset_obj,
                 tokenizer: transformers.AutoTokenizer.from_pretrained,
                 ):
        """
          Initializes the NegativeSamplingDataModule.

          Args:
              train_data: Training data examples.
              test_data: Test data examples.
              dev_data: Development/validation data examples.
              config: Configuration object.
              dataset_obj: Dataset class for creating training, dev, and test datasets.
              tokenizer: Tokenizer for processing text.
          """
        super().__init__()
        self.config = config
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.dataset_obj = dataset_obj
        self.tokenizer = tokenizer
        self.dataset = {}

    def setup(self, stage=None) -> None:
        """
        Initializes datasets based on the provided data and negative samples.

        Args:
            stage: The stage of training (None for the default case).

        Returns:
            None
        """
        self.dataset["train_dataset"] = self.dataset_obj(
            data=self.train_data,
            tokenizer=self.tokenizer,
            max_len=self.config.max_length)
        self.dataset["dev_dataset"] = self.dataset_obj(
            data=self.dev_data,
            tokenizer=self.tokenizer,
            max_len=self.config.max_length)
        self.dataset["test_dataset"] = self.dataset_obj(
            data=self.test_data,
            tokenizer=self.tokenizer,
            max_len=self.config.max_length)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return torch.utils.data.DataLoader(self.dataset["train_dataset"],
                                           batch_size=self.config.per_device_train_batch_size,
                                           num_workers=self.config.num_workers,
                                           shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return torch.utils.data.DataLoader(self.dataset["dev_dataset"],
                                           batch_size=self.config.per_device_train_batch_size,
                                           num_workers=self.config.num_workers)

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return torch.utils.data.DataLoader(self.dataset["test_dataset"],
                                           batch_size=self.config.per_device_train_batch_size,
                                           num_workers=self.config.num_workers)
