"""
    Generative_AI_Authorship_Verification Project:
        data_preparation:
            data_preparation.py
"""


# ============================ Third Party libs ============================

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
    samples = []
    human_data = type_2_data.pop("human")  # Remove "human" key from type_2_data and get its value
    for key, value in type_2_data.items():
        for index, data in enumerate(value):
            human_text = human_data[index]["text"]
            data_text = data["text"]

            if index % 2 == 0:
                sample = {"text1": human_text, "text2": data_text, "label": 1, "author": key}
            else:
                sample = {"text1": data_text, "text2": human_text, "label": 0, "author": key}
            samples.append(sample)
    return samples
