"""
    Generative_AI_Authorship_Verification Project:
        src:
            binoculars_runner.py
"""
# ============================ Third Party libs =======================
import os
from sklearn.metrics import accuracy_score
# ============================ My packages ============================
from src.binoculars import Binoculars
from src.configuration import BaseConfig
from src.data_loader import load_jsonl

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))

    BINO = Binoculars()

    RESULTS = []
    TRUE_LABELS = []
    print(len(DATA))
    for index, sample in enumerate(DATA):
        print(index)
        TEXT1_SCORE = BINO.compute_score(sample["text1"])
        TEXT2_SCORE = BINO.compute_score(sample["text2"])
        if TEXT1_SCORE < TEXT2_SCORE:
            RESULTS.append(0)
        else:
            RESULTS.append(1)
        TRUE_LABELS.append(sample["label"] == 0)

    ACC = accuracy_score(y_true=TRUE_LABELS, y_pred=RESULTS)
    print("ACC", ACC)
