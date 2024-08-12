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
from src.inference import runner_factory

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    DEV_DATA = load_jsonl(ARGS.inputDataset)

    DEV_DATASET = Dataset.from_list(DEV_DATA)

    RESULTS = runner_factory(samples=DEV_DATASET, inference_methods=["llama", "mistral",
                                                                     "binocular"],
                             observer_name_or_path=ARGS.observer_name_or_path,
                             performer_name_or_path=ARGS.performer_name_or_path,
                             llama_model_path=ARGS.llama_model_path,
                             llama_peft_model_path=ARGS.llama_peft_model_path,
                             mistral_model_path=ARGS.mistral_model_path,
                             mistral_peft_model_path=ARGS.mistral_peft_model_path,
                             max_length=ARGS.max_length,
                             mode="low-fpr",
                             binoculars_accuracy_threshold=ARGS.binoculars_accuracy_threshold,
                             binoculars_fpr_threshold=ARGS.binoculars_fpr_threshold,
                             device=ARGS.device
                             )

    OUTPUTS = []
    for index, sample in enumerate(DEV_DATASET):
        OUTPUTS.append({"id": sample["id"],
                        "is_human": RESULTS[index]})

    save_to_jsonl(data=OUTPUTS, file_path=ARGS.outputDir)
