"""
    Generative_AI_Authorship_Verification Project:
        data_preparation:
            data_preparation.py
"""

# ============================ Third Party libs ============================
import random


def create_samples(type_2_data: dict):
    """
    Create samples for training data.

    This function generates samples for training data from a dictionary containing
    human and type 2 data. Each sample consists of two texts, a label indicating
    whether the texts are associated or not, and the author's key.

    Parameters:
        type_2_data (dict): A dictionary containing human and type 2 data.

    Returns:
        list: A list of samples. Each sample is a dictionary with keys 'text1', 'text2',
              'label', and 'author'.
    """
    pair_samples = []
    human_data = type_2_data.pop("human")  # Remove "human" key from type_2_data and get its value
    machine_names = list(type_2_data.keys())
    for index, item in enumerate(human_data):
        selected_key = random.choice(machine_names)
        if index % 2 == 0:
            pair_sample = {"text1": item["text"], "text2": type_2_data[selected_key][index]["text"],
                           "label": 1,
                           "author": selected_key}
        else:
            pair_sample = {"text1": type_2_data[selected_key][index]["text"], "text2": item["text"],
                           "label": 0,
                           "author": selected_key}
        pair_samples.append(pair_sample)
    return pair_samples


def create_single_samples(pair_samples):
    """
    Create single samples from pairs of samples.

    Args:
        pair_samples (list of dict): List of dictionaries representing pairs of samples.
            Each dictionary should have keys "text1", "text2", and "label".

    Returns:
        list of dict: List of dictionaries representing single samples.
            Each dictionary has keys "text" and "label".
    """
    samples = []
    for item in pair_samples:
        if item["label"] == 0:
            samples.append({"text": item["text1"], "label": "machine", "author": item["author"]})
            samples.append({"text": item["text2"], "label": "human", "author": "human"})
        elif item["label"] == 1:
            samples.append({"text": item["text1"], "label": "human", "author": "human"})
            samples.append({"text": item["text2"], "label": "machine", "author": item["author"]})
    return samples


# def create_samples(type_2_data: dict):
#     """
#     Create samples for training data.
#
#     This function generates samples for training data from a dictionary containing
#     human and type 2 data. Each sample consists of two texts, a label indicating
#     whether the texts are associated or not, and the author's key.
#
#     Parameters:
#         type_2_data (dict): A dictionary containing human and type 2 data.
#
#     Returns:
#         list: A list of samples. Each sample is a dictionary with keys 'text1', 'text2',
#               'label', and 'author'.
#     """
#     samples = []
#     human_data = type_2_data.pop("human")  # Remove "human" key from type_2_data and get its value
#     for key, value in type_2_data.items():
#         for index, data in enumerate(value):
#             human_text = human_data[index]["text"]
#             data_text = data["text"]
#
#             if index % 2 == 0:
#                 sample = {"text1": human_text, "text2": data_text, "label": 1, "author": key}
#             else:
#                 sample = {"text1": data_text, "text2": human_text, "label": 0, "author": key}
#             samples.append(sample)
#     return samples


def sequence_classification_data_creator(data: list) -> (list, dict, dict):
    """
    Create sequence classification data by concatenating two texts with a separator.

    Args:
        data (list): A list of objects containing 'text1', 'text2', and 'label' attributes.

    Returns:
        tuple: A tuple containing three elements:
            - list: A list of dictionaries, each containing 'text' (concatenated text) and 'label'.
            - dict: A dictionary mapping labels to their corresponding ids.
            - dict: A dictionary mapping ids to their corresponding labels.
    """
    samples = []
    labels = set()
    for sample in data:
        samples.append(
            {"text": sample["text1"] + "[SEP]" + sample["text2"], "labels": sample["label"]})
        labels.add(sample["label"])

    # Creating label to id and id to label mappings
    label2id = {label: idx for idx, label in enumerate(sorted(labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    return samples, label2id, id2label
