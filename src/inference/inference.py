"""
    Generative_AI_Authorship_Verification Project:
        inference:
            inference.py
"""

# ============================ Third Party libs ============================
from abc import ABC, abstractmethod

import torch
import tqdm
from peft import PeftModel
from transformers import LlamaForSequenceClassification, AutoTokenizer, \
    AutoModelForSequenceClassification


class Inference(ABC):
    def __init__(self, model_path, max_length, device):
        self.model_path = model_path
        self.max_length = max_length
        self.device = device

        self.tokenizer = None
        self.model = None

    @abstractmethod
    def load_model(self):
        """

        Returns:

        """

    @abstractmethod
    def runner(self, samples):
        """

        Returns:

        """

    def add_pad_token(self):
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenizer_wrapper(self, input_text):
        return self.tokenizer(input_text, truncation=True,
                              padding=True,
                              max_length=self.max_length,
                              return_tensors="pt")

    def inferencer_wrapper(self, sample):
        output_1 = torch.softmax(
            self.model(**self.tokenizer_wrapper(sample["text1"])).logits, dim=1
        ).detach().cpu().numpy()[0][0]
        output_2 = torch.softmax(
            self.model(**self.tokenizer_wrapper(sample["text2"])).logits, dim=1
        ).detach().cpu().numpy()[0][0]
        return 0 if output_1 - output_2 > 0.5 else 1


class LLamaInferencer(Inference):
    def __init__(self, model_path, peft_model_path, max_length, device):
        super().__init__(model_path, max_length, device)
        self.peft_model_path = peft_model_path

        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.add_pad_token()

        label2id = {"human": 1, "machine": 0}
        id2label = {idx: label for label, idx in label2id.items()}
        base_model = LlamaForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            device_map=self.device
        )

        self.model = PeftModel.from_pretrained(base_model, self.peft_model_path)
        self.model.eval()

    def runner(self, samples):
        results = []
        for sample in tqdm.tqdm(samples):
            results.append(self.inferencer_wrapper(sample))
        return results


class MistralInferencer(Inference):
    def __init__(self, model_path, peft_model_path, max_length, device):
        super().__init__(model_path, max_length, device)
        self.peft_model_path = peft_model_path

        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.add_pad_token()

        label2id = {"human": 1, "machine": 0}
        id2label = {idx: label for label, idx in label2id.items()}
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            device_map=self.device
        )

        self.model = PeftModel.from_pretrained(base_model, self.peft_model_path)
        self.model.eval()

    def runner(self, samples):
        results = []
        for sample in tqdm.tqdm(samples):
            results.append(self.inferencer_wrapper(sample))
        return results


