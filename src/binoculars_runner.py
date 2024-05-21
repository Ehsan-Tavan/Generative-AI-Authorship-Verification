"""
    Generative_AI_Authorship_Verification Project:
        src:
            binoculars_runner.py
"""
# ============================ Third Party libs =======================
import os
from sklearn.metrics import accuracy_score
from transformers import BartTokenizer

# ============================ My packages ============================
from src.binoculars import Binoculars
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.models.bart_paraphraser import BartParaphraser


def paraphraser(text):
    input_sequences_encoding = TOKENIZER.encode_plus(
        text=text,
        max_length=ARGS.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt")
    results = PL_MODEL.generate(
        input_sequences_encoding.input_ids.to(ARGS.device),
        num_beams=10, num_return_sequences=10, early_stopping=True,
        no_repeat_ngram_size=3, max_length=ARGS.max_length,
        remove_invalid_values=True,
    )
    paraphrased_text = [TOKENIZER.decode(res, skip_special_tokens=True)
                        for res in results]
    return paraphrased_text[0]


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))

    PL_MODEL = BartParaphraser.load_from_checkpoint("/mnt/disk2/ehsan.tavan/gen_ai/assets/"
                                                    "saved_model/Paraphraser_Bart/version_0/"
                                                    "checkpoints/"
                                                    "QTag-epoch=04-val_loss=1.41.ckpt").model
    TOKENIZER = BartTokenizer.from_pretrained(ARGS.bart_model_path)

    BINO = Binoculars()

    RESULTS = []
    TRUE_LABELS = []
    for index, sample in enumerate(DATA):
        TEXT1_SCORE = BINO.compute_score(sample["text1"])
        TEXT2_SCORE = BINO.compute_score(sample["text2"])

        paraphraser_text1 = paraphraser(sample["text1"])
        PARAPHRASER_TEXT1_SCORE = BINO.compute_score(paraphraser_text1)

        paraphraser_text2 = paraphraser(sample["text2"])
        PARAPHRASER_TEXT2_SCORE = BINO.compute_score(paraphraser_text2)

        print(TEXT1_SCORE)
        print(PARAPHRASER_TEXT1_SCORE)
        print("###############")
        print(TEXT2_SCORE)
        print(PARAPHRASER_TEXT2_SCORE)

        if TEXT1_SCORE < TEXT2_SCORE:
            RESULTS.append(0)
        else:
            RESULTS.append(1)
        TRUE_LABELS.append(sample["label"] == 0)

        print("Ture label: ", sample["label"])
        print("Ture label: ", RESULTS[index])

    ACC = accuracy_score(y_true=TRUE_LABELS, y_pred=RESULTS)
    print("ACC", ACC)
