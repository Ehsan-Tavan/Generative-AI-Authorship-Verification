"""
    Generative_AI_Authorship_Verification Project:
"""

# ============================ Third Party libs =======================
import os
from sklearn.metrics import accuracy_score
import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, \
    LlamaForSequenceClassification
from huggingface_hub import notebook_login
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.data_preparation import sequence_classification_data_creator

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    notebook_login()

    DEV_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.single_dev_file))
    DEV_SAMPLES, _, _ = sequence_classification_data_creator(DEV_DATA)
    DEV_DATASET = Dataset.from_list(DEV_SAMPLES)

    TOKENIZER = AutoTokenizer.from_pretrained(ARGS.llama_model_path, trust_remote_code=True)
    # DM added
    if TOKENIZER.pad_token is None:
        if TOKENIZER.eos_token is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        else:
            TOKENIZER.add_special_tokens({"pad_token": "[PAD]"})

    LABEL2ID = {"human": 1, "machine": 0}
    ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

    TRUE_LABELS = [sample["labels"] for sample in DEV_DATASET]

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=ARGS.load_in_8bit,  # load model in 8-bit precision
        load_in_4bit=ARGS.load_in_4bit,  # load model in 4-bit precision
        bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
        bnb_4bit_use_double_quant=ARGS.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # During computation, pre-trained model should be loaded in BF16 format
    )

    BASE_MODEL = LlamaForSequenceClassification.from_pretrained(
        ARGS.llama_model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        device_map=ARGS.device
    )

    PEFT_MODEL = PeftModel.from_pretrained(
        BASE_MODEL, ARGS.llama_peft_model_path)
    PEFT_MODEL.eval()
    PEFT_MODEL.print_trainable_parameters()
    PEFT_MODEL.push_to_hub("Generative-AV-PAN/llama-2-7b")
    ss

    RESULTS = []

    for index, sample in enumerate(DEV_DATASET):
        print(index / len(DEV_DATASET) * 100)
        tokenized_text = TOKENIZER(sample["text"], truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors="pt").to(ARGS.device)

        output = PEFT_MODEL(**tokenized_text).logits
        output = torch.argmax(output, dim=1).detach().cpu().numpy()
        print(output)
        RESULTS.extend(output)
        print(RESULTS)

    print(RESULTS)
    ACC = accuracy_score(y_true=TRUE_LABELS, y_pred=RESULTS)
    print("ACC", ACC)
