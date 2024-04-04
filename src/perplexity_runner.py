"""
    Generative_AI_Authorship_Verification Project:
        src:
            perplexity_runner.py
"""
# ============================ Third Party libs =======================
import os

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.models import PerplexityClassifier

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.dev_file))[:10]
    perplexity_classifier = PerplexityClassifier(ARGS)

    ACC, PREDICTED_LABELS = perplexity_classifier.evaluate(DEV_DATA)
    print("ACC", ACC)
