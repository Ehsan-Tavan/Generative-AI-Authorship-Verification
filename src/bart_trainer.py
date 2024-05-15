"""
    Generative_AI_Authorship_Verification Project:
        src:
            bart_trainer.py
"""
# ============================ Third Party libs =======================
import os
import torch

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_preparation import prepare_data, create_paraphraser_data
from src.dataset import BARTDataset
from src.models.bart_paraphraser import BartParaphraser

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TRAIN_DATA, DEV_DATA = prepare_data(
        pair_train_data_path=os.path.join(ARGS.processed_data_dir, ARGS.pair_train_file),
        pair_dev_data_path=os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file),
        single_train_data_path=os.path.join(ARGS.processed_data_dir, ARGS.single_train_file),
        single_dev_data_path=os.path.join(ARGS.processed_data_dir, ARGS.single_dev_file),
        training_data_type=ARGS.training_data_type)
    print(TRAIN_DATA[0])

    DATA_TRAIN = create_paraphraser_data(TRAIN_DATA)
    DATA_DEV = create_paraphraser_data(DEV_DATA)

    MODEL = BartParaphraser(lm_model_path=ARGS.bart_model_path,
                            learning_rate=ARGS.learning_rate,
                            dataset_obj=BARTDataset,
                            train_data=DATA_TRAIN,
                            dev_data=DATA_DEV,
                            args=ARGS)

    MODEL.fit(check_point_monitor="val_loss", check_point_mode="min",
              early_stopping_monitor="val_loss", early_stopping_patience=5)
