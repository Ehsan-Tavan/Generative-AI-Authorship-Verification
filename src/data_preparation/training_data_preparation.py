"""
    Generative_AI_Authorship_Verification Project:
        data_preparation:
            training_data_preparation.py
"""
# ============================ Third Party libs =======================
from abc import ABC, abstractmethod

# ============================ My packages ============================
from src.data_loader import read_csv, load_jsonl


class DataProcessor(ABC):
    """
    Abstract base class for data processors.
    """

    @abstractmethod
    def prepare_data(self, train_data_path: str, dev_data_path: str):
        """
        Prepare training and validation data.

        Args:
            train_data_path (str): Path to the training data.
            dev_data_path (str): Path to the validation data.

        Returns:
            tuple: A tuple containing the prepared training data and validation data.
        """
        pass


class SingleTextDataProcessor(DataProcessor):
    """
    Data processor for single text samples.
    """

    def prepare_data(self, train_data_path: str, dev_data_path: str):
        """
        Prepare training and validation data for single text samples.

        Args:
            train_data_path (str): Path to the training data.
            dev_data_path (str): Path to the validation data.

        Returns:
            tuple: A tuple containing the prepared training data and validation data.
        """

        train_data = self._read_data(train_data_path)
        dev_data = self._read_data(dev_data_path)

        train_data = self._process_data(train_data)
        dev_data = self._process_data(dev_data)

        print(f"We have {len(train_data[0])} training samples.")
        print(f"We have {len(dev_data[0])} validation samples.")
        return train_data, dev_data

    @staticmethod
    def _read_data(data_path):
        """
        Read data from a CSV file.

        Args:
            data_path (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: The data read from the CSV file.
        """

        data = read_csv(data_path)
        return data.dropna()

    @staticmethod
    def _process_data(data):
        """
        Process the data by mapping labels and converting to lists.

        Args:
            data (pandas.DataFrame): Input data.

        Returns:
            list: Processed data as a list containing text and labels.
        """
        label_dict = {"human": 1, "machine": 0}
        data["label"] = data["label"].map(lambda x: label_dict[x])
        return [list(data["text"]), list(data["label"])]


class TextPairDataProcessor(DataProcessor):
    """
    Data processor for text pair samples.
    """

    def prepare_data(self, train_data_path: str, dev_data_path: str):
        """
         Prepare training and validation data for text pair samples.

         Args:
             train_data_path (str): Path to the training data.
             dev_data_path (str): Path to the validation data.

         Returns:
             tuple: A tuple containing the prepared training data and validation data.
         """
        train_data = self._load_data(train_data_path)
        dev_data = self._load_data(dev_data_path)

        print(f"We have {len(train_data)} training samples.")
        print(f"We have {len(dev_data)} validation samples.")
        return train_data, dev_data

    @staticmethod
    def _load_data(data_path: str):
        """
         Load data from a JSONL file.

         Args:
             data_path (str): Path to the JSONL file.

         Returns:
             list: The data loaded from the JSONL file.
         """
        return load_jsonl(data_path)


class DataProcessorFactory:
    """
    Factory class to create data processors based on the type of training data.
    """

    @staticmethod
    def create_data_processor(training_data_type: str):
        """
        Create a data processor based on the type of training data.

        Args:
            training_data_type (str): Type of training data ("single_text" or "text_pair").

        Returns:
            DataProcessor: An instance of the appropriate data processor.
        """
        if training_data_type == "single_text":
            return SingleTextDataProcessor()
        elif training_data_type == "text_pair":
            return TextPairDataProcessor()
        else:
            raise ValueError("Invalid training_data_type")


def prepare_data(train_data_path: str, dev_data_path: str, training_data_type: str):
    """
    Prepare training and validation data based on the type of training data.

    Args:
        train_data_path (str): Path to the training data.
        dev_data_path (str): Path to the validation data.
        training_data_type (str): Type of training data ("single_text" or "text_pair").

    Returns:
        tuple: A tuple containing the prepared training data and validation data.
    """
    data_processor = DataProcessorFactory.create_data_processor(training_data_type)
    return data_processor.prepare_data(train_data_path, dev_data_path)
