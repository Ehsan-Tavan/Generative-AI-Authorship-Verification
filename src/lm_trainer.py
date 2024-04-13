"""
    Generative_AI_Authorship_Verification Project:
        src:
            lm_trainer.py
"""
# ============================ Third Party libs =======================
import os

import torch

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import read_csv
from src.models.lm_classifier import LmClassifier

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    TRAIN_DATA = read_csv(os.path.join(ARGS.processed_data_dir, ARGS.train_file))
    DEV_DATA = read_csv(os.path.join(ARGS.processed_data_dir, ARGS.dev_file))

    TRAIN_DATA = TRAIN_DATA.dropna()
    DEV_DATA = DEV_DATA.dropna()

    label_dict = {"human": 1, "machine": 0}

    TRAIN_DATA.label = TRAIN_DATA.label.map(lambda x: label_dict[x])
    DEV_DATA.label = DEV_DATA.label.map(lambda x: label_dict[x])

    TRAIN_DATA = [list(TRAIN_DATA["text"]), list(TRAIN_DATA["label"])]
    DEV_DATA = [list(DEV_DATA["text"]), list(DEV_DATA["label"])]

    print(f"We have {len(TRAIN_DATA[0])} training samples.")
    print(f"We have {len(DEV_DATA[0])} validation samples.")

    MODEL = LmClassifier(model_path=ARGS.model_path, args=ARGS,
                         loss_fct=torch.nn.CrossEntropyLoss(),
                         optimizer_class=torch.optim.AdamW,
                         train_data=TRAIN_DATA,
                         dev_data=DEV_DATA, pooling_methods=["mean"],
                         optimizer_params={"lr": ARGS.learning_rate})

    MODEL.fit()