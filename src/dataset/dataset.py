"""
    Generative_AI_Authorship_Verification Project:
        dataset:
            dataset.py
"""
# ============================ Third Party libs ============================
from abc import ABC

import torch


class AbstractDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for creating PyTorch datasets.
    """

    def __init__(self,
                 data: list,
                 tokenizer,
                 max_len: int = None):
        """
        Initialize the AbstractDataset.

        Args:
            data: List of data samples.
            tokenizer: Tokenizer for processing text.
            max_len: Maximum length of the input sequences.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _encode_sample(self, text, text_pair=None):
        """
        Encode a text sample using the provided tokenizer.

        Args:
            text: Input text to be encoded.

        Returns:
            dict: Dictionary containing the encoded input_ids and attention_mask.
        """
        encoded_sample = self.tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
            max_length=self.max_len,
            return_tensors="pt",
            padding="max_length",
            truncation="longest_first"
        )

        # Flatten the tensors for easy access
        input_ids = encoded_sample["input_ids"].flatten()
        attention_mask = encoded_sample["attention_mask"].flatten()

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    @staticmethod
    def __getitem__(self, item_index):
        """
         Abstract method for retrieving a sample from the dataset.

         Args:
             item_index: Index of the sample to retrieve.

         Returns:
             dict: Sample data.
         """


class TextPairDataset(AbstractDataset):
    def __getitem__(self,
                    item_index: int) -> dict:
        """
        Retrieve a sample from the dataset.

        Args:
            item_index: Index of the sample to retrieve.

        Returns:
            dict: Sample data.
        """
        sample = self.data[item_index]
        self.target = sample["label"]

        # Encode the text using the tokenizer
        encoded_text = self._encode_sample(text=sample["text1"], text_pair=sample["text2"])

        encoded_text["targets"] = self.target

        return encoded_text


class SingleTextDataset(AbstractDataset):
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self,
                    item_index: int) -> dict:
        """
        Retrieve a sample from the dataset.

        Args:
            item_index: Index of the sample to retrieve.

        Returns:
            dict: Sample data.
        """
        self.text = self.data[item_index]["text"]
        self.target = self.data[item_index]["label"]

        # Encode the text using the tokenizer
        encoded_text = self._encode_sample(self.text)

        encoded_text["targets"] = self.target

        return encoded_text
