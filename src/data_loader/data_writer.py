"""
    Generative_AI_Authorship_Verification Project:
        data_loader:
            data_writer.py
"""

# ============================ Third Party libs ============================
import json


def save_to_jsonl(data: list, file_path: str):
    """
    Save data to a JSONL (JSON Lines) file.

    Parameters:
        data (list): The list of data to be saved.
        file_path (str): The path to save the JSONL file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False) + '\n'
            file.write(json_line)
