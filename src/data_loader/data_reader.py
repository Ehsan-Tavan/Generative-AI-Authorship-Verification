"""
    Generative_AI_Authorship_Verification Project:
        data_loader:
            data_reader.py
"""
# ============================ Third Party libs ============================
import json
import pandas as pd


def load_jsonl(file_path: str) -> list:
    """
    Load data from a JSONL (JSON Lines) file.

    Parameters:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list containing parsed JSON objects from the file.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def read_csv(path: str, columns: list = None, names: list = None) -> pd.DataFrame:
    """
    read_csv function for reading csv files

    Args:
        path:
        columns:
        names:

    Returns:

    """
    dataframe = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    return dataframe.rename(columns=dict(zip(columns, names))) if names else dataframe

