"""
    Generative_AI_Authorship_Verification Project:
        src:
            paraphraser_inferencer.py
"""

# ============================ Third Party libs =======================
import os

import torch
from transformers import BartTokenizer

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_preparation import prepare_data, create_paraphraser_data
from src.dataset import BARTDataset
from src.models.bart_paraphraser import BartParaphraser

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    PL_MODEL = BartParaphraser.load_from_checkpoint("/mnt/disk2/ehsan.tavan/gen_ai/assets/"
                                                    "saved_model/Paraphraser_Bart/version_0/"
                                                    "checkpoints/"
                                                    "QTag-epoch=04-val_loss=1.41.ckpt").model.to("cuda:0")

    TRAIN_DATA, DEV_DATA = prepare_data(
        pair_train_data_path=os.path.join(ARGS.processed_data_dir, ARGS.pair_train_file),
        pair_dev_data_path=os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file),
        single_train_data_path=os.path.join(ARGS.processed_data_dir, ARGS.single_train_file),
        single_dev_data_path=os.path.join(ARGS.processed_data_dir, ARGS.single_dev_file),
        training_data_type=ARGS.training_data_type)

    DATA_TRAIN = create_paraphraser_data(TRAIN_DATA)
    DATA_DEV = create_paraphraser_data(DEV_DATA)

    TOKENIZER = BartTokenizer.from_pretrained(ARGS.bart_model_path)

    DATASET = BARTDataset(data=DATA_DEV, tokenizer=TOKENIZER,
                          max_len=ARGS.max_length)
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(DATASET, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data in enumerate(DEV_DATA):
            print(data)
            INPUT_SEQUENCES_ENCODING = TOKENIZER.encode_plus(text=data[1]["text1"],
                                                             max_length=ARGS.max_length,
                                                             padding="max_length", truncation=True,
                                                             return_tensors="pt")

            RESULTS = PL_MODEL.generate(INPUT_SEQUENCES_ENCODING.input_ids.to("cuda:0"),
                                        num_beams=10, num_return_sequences=10, early_stopping=True,
                                        no_repeat_ngram_size=3, max_length=ARGS.max_length,
                                        remove_invalid_values=True,
                                        )
            print("EEEeehsaaaannnnnnn")
            RESULTS = [TOKENIZER.decode(res, skip_special_tokens=True) for res in RESULTS]
            print("generated similar questions are : \n")
            print("=" * 40)
            print(RESULTS)

