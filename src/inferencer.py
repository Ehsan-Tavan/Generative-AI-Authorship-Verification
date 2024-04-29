"""
    Generative_AI_Authorship_Verification Project:
        src:
            inferencer.py
"""
# ============================ Third Party libs =======================

from datasets import Dataset

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl, save_to_jsonl
from src.inference import MistralInferencer

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    # DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_dev_file))
    DEV_DATA = load_jsonl("/mnt/disk2/ehsan.tavan/gen_ai/data/Processed/dev_data_1.jsonl")

    # DEV_SAMPLES, _, _ = sequence_classification_data_creator2(DEV_DATA)
    DEV_DATASET = Dataset.from_list(DEV_DATA)
    # TRUE_LABELS = [sample["labels"] for sample in DEV_DATASET]

    INFERENCER = MistralInferencer(model_path=ARGS.model_path,
                                   peft_model_path="/mnt/disk2/ehsan.tavan/gen_ai/assets/"
                                                   "saved_model/"
                                                   "Generative_AI_Authorship_Verification_mistral/"
                                                   "version_1/checkpoint-868",
                                   max_length=ARGS.max_length,
                                   device=ARGS.device)

    RESULTS = INFERENCER.runner(DEV_DATASET)

    OUTPUTS = []
    for index, sample in enumerate(DEV_DATASET):
        OUTPUTS.append({"id": sample["id"],
                        "is_human": RESULTS["index"]})

    print(OUTPUTS)
    save_to_jsonl(data=OUTPUTS, file_path="output.jsonl")
    # ACC = accuracy_score(y_true=TRUE_LABELS, y_pred=RESULTS)
    # print("ACC", ACC)
