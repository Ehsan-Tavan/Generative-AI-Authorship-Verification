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


def sequence_classification_data_creator2(data: list) -> (list, dict, dict):
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
    label2id = {"human": 1, "machine": 0}
    for sample in data:
        samples.append(
            {"text1": sample["text1"], "text2": sample["text2"],
             "labels": sample["label"]})

    # Creating label to id and id to label mappings
    id2label = {idx: label for label, idx in label2id.items()}
    return samples, label2id, id2label


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
    label2id = {"human": 1, "machine": 0}
    for sample in data:
        samples.append(
            {"text": sample["text"], "labels": label2id[sample["label"]]})

    # Creating label to id and id to label mappings
    id2label = {idx: label for label, idx in label2id.items()}
    return samples, label2id, id2label


def paraphraser_data_creator(data: list, mode="train") -> (list, dict, dict):
    instructions = []
    instruction_key = "### Instruction: "
    task_instruction = "You are a model designed for paraphrasing. Please rephrase the given " \
                       "text in a natural, human-like style. " \
                       "Ensure that the generated text does not exceed " \
                       "the length of the input text."
    input_key = "### Text: "
    end_key = "### End"
    response_key = "### Response: "

    for sample in data:
        instruction = f"{instruction_key}\n{task_instruction}"
        input_text = f"{input_key}\n{sample['text1']}"
        response = f"{response_key}\n{sample['text2']}"

        end = f"{end_key}"
        if mode == "train":
            parts = [part for part in
                     [instruction, input_text, response, end]]
        else:
            parts = [part for part in
                     [instruction, input_text, response_key]]

        formatted_prompt = "\n".join(parts)
        instructions.append({"instruction": formatted_prompt})

    return instructions


def generation_data_creator(data: list, mode="train") -> (list, dict, dict):
    instructions = []
    instruction_key = "### Instruction: "
    task_instruction = "Classify input text into human_generated or ai_generated text."
    input_key = "### Text: "
    end_key = "### End"
    response_key = "### Response: "

    for sample in data:
        instruction = f"{instruction_key}\n{task_instruction}"
        input_text = f"{input_key}\n{sample['text']}"
        if sample["label"] == 0:
            response = f"{response_key}\nai_generated"

        else:
            response = f"{response_key}\nhuman_generated"

        end = f"{end_key}"
        if mode == "train":
            parts = [part for part in
                     [instruction, input_text, response, end]]
        else:
            parts = [part for part in
                     [instruction, input_text, response_key]]

        formatted_prompt = "\n".join(parts)
        instructions.append({"instruction": formatted_prompt})

    return instructions


def instruction_tuning_data_creator(data: list, mode="train") -> (list, dict, dict):
    instructions = []
    instruction_key = "### Instruction: "
    task_instruction = "I give you two texts and ask you to determine which one is authored by " \
                       "humans and which one is authored by machines. Your output is just a 0 " \
                       "and 1 and do not generate anything else. 0 means text_1 is authored " \
                       "by the machine and 1 means text_2 is authored by the machine."
    input_key1 = "### Text1: "
    input_key2 = "### Text2: "
    end_key = "### End"
    response_key = "### Response: "

    for sample in data:
        instruction = f"{instruction_key}\n{task_instruction}"
        input_text = f"{input_key1}\n{sample['text1']}\n{input_key2}{sample['text2']}"
        end = f"{end_key}"
        if mode == "train":
            response = f"{response_key}\n{sample['label']}"
            parts = [part for part in
                     [instruction, input_text, response, end]]
        else:
            parts = [part for part in
                     [instruction, input_text, response_key]]

        formatted_prompt = "\n".join(parts)
        instructions.append({"instruction": formatted_prompt})
    return instructions
