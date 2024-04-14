"""
    Generative_AI_Authorship_Verification Project:
        runner.py
"""

# ============================ Third Party libs =======================
import os

from sklearn.model_selection import train_test_split

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl, save_to_jsonl
from src.data_preparation import create_samples, create_single_samples

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    TYPE_2_DATA = {}

    for filename in os.listdir(ARGS.raw_data_dir):
        if filename == "human.jsonl":
            TYPE_2_DATA["human"] = load_jsonl(os.path.join(ARGS.raw_data_dir, "human.jsonl"))
        elif filename == "machines":
            for machine_generated_filename in os.listdir(os.path.join(ARGS.raw_data_dir, filename)):
                if machine_generated_filename.endswith('.jsonl'):
                    TYPE_2_DATA[machine_generated_filename[:-6]] = load_jsonl(
                        os.path.join(ARGS.raw_data_dir, filename, machine_generated_filename))

    SAMPLES = create_samples(TYPE_2_DATA)

    TRAIN_DATA, DEV_DATA = train_test_split(SAMPLES, test_size=0.2, random_state=1234)

    SINGLE_TRAIN_DATA = create_single_samples(TRAIN_DATA)
    SINGLE_DEV_DATA = create_single_samples(DEV_DATA)

    print(f"We have {len(TRAIN_DATA)}  pair samples train data")
    print(f"We have {len(DEV_DATA)} pair samples dev data")

    print(f"We have {len(SINGLE_TRAIN_DATA)}  single samples train data")
    print(f"We have {len(SINGLE_DEV_DATA)} single samples dev data")

    save_to_jsonl(data=TRAIN_DATA,
                  file_path=os.path.join(ARGS.processed_data_dir, ARGS.pair_train_file))
    save_to_jsonl(data=DEV_DATA,
                  file_path=os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))

    save_to_jsonl(data=SINGLE_TRAIN_DATA,
                  file_path=os.path.join(ARGS.processed_data_dir, ARGS.single_train_file))
    save_to_jsonl(data=SINGLE_DEV_DATA,
                  file_path=os.path.join(ARGS.processed_data_dir, ARGS.single_dev_file))
