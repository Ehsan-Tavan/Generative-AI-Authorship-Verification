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
from src.data_preparation import prepare_data
from src.dataset import SingleTextDataset
from src.models.lm_classifier import LmClassifier


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TRAIN_DATA, DEV_DATA = prepare_data(
        train_data_path=os.path.join(ARGS.processed_data_dir, ARGS.train_file),
        dev_data_path=os.path.join(ARGS.processed_data_dir, ARGS.dev_file),
        training_data_type=ARGS.training_data_type)

    MODEL = LmClassifier(model_path=ARGS.model_path, args=ARGS,
                         loss_fct=torch.nn.CrossEntropyLoss(),
                         optimizer_class=torch.optim.AdamW,
                         train_data=TRAIN_DATA,
                         dataset_obj=SingleTextDataset,
                         dev_data=DEV_DATA, pooling_methods=["mean"],
                         optimizer_params={"lr": ARGS.learning_rate})

    MODEL.fit(check_point_monitor="dev_loss", check_point_mode="min",
              early_stopping_monitor="dev_loss", early_stopping_patience=5)
