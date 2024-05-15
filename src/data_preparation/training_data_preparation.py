"""
    Generative_AI_Authorship_Verification Project:
        data_preparation:
            training_data_preparation.py
"""
# ============================ Third Party libs =======================
from abc import ABC, abstractmethod

# ============================ My packages ============================
from src.data_loader import load_jsonl


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

        train_data = self._load_data(train_data_path)
        dev_data = self._load_data(dev_data_path)

        train_data = self._process_data(train_data)
        dev_data = self._process_data(dev_data)

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

    @staticmethod
    def _process_data(data):
        """
        Process the data by mapping labels and converting to lists.

        Args:
            data (list of dict): Input data.

        Returns:
            list: Processed data as a list containing text and labels.
        """
        label_dict = {"human": 1, "machine": 0}
        for sample in data:
            sample["label"] = label_dict[sample["label"]]
        return data


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


def prepare_data(pair_train_data_path: str, pair_dev_data_path: str,
                 single_train_data_path: str, single_dev_data_path: str,
                 training_data_type: str):
    if training_data_type == "single_text":
        train_data_path = single_train_data_path
        dev_data_path = single_dev_data_path
    elif training_data_type == "text_pair":
        train_data_path = pair_train_data_path
        dev_data_path = pair_dev_data_path
    else:
        raise ValueError("Invalid training_data_type")

    data_processor = DataProcessorFactory.create_data_processor(training_data_type)
    return data_processor.prepare_data(train_data_path, dev_data_path)


def create_paraphraser_data(data):
    output_data = {"text1": [], "text2": [], "label": []}

    # Iterate over each dictionary in the list
    for item in data:
        for key in item:
            if key in output_data:
                output_data[key].append(item[key])
    return output_data
