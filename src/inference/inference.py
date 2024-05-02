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
    AutoModelForSequenceClassification, AutoModelForCausalLM

# ============================ My packages ============================
from src.utils import perplexity, entropy, calculate_mean


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
                              return_tensors="pt").to(self.device)

    def inferencer_wrapper(self, sample):
        output_1 = torch.softmax(
            self.model(**self.tokenizer_wrapper(sample["text1"])).logits, dim=1
        ).detach().cpu().numpy()[0][0]
        output_2 = torch.softmax(
            self.model(**self.tokenizer_wrapper(sample["text2"])).logits, dim=1
        ).detach().cpu().numpy()[0][0]
        return 0 if output_1 - output_2 > 0.5 else 1


class LLamaInferencer(Inference):
    def __init__(self, **kwargs):
        super().__init__(kwargs["llama_model_path"], kwargs["max_length"], kwargs["device"])
        self.peft_model_path = kwargs["llama_peft_model_path"]

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
    def __init__(self, **kwargs):
        super().__init__(kwargs["mistral_model_path"], kwargs["max_length"], kwargs["device"])
        self.peft_model_path = kwargs["mistral_peft_model_path"]

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


class BinocularInferencer(Inference):
    def __init__(self, **kwargs):
        super().__init__(kwargs["observer_name_or_path"], kwargs["max_length"], kwargs["device"])
        self.observer_name_or_path = kwargs["observer_name_or_path"]
        self.performer_name_or_path = kwargs["performer_name_or_path"]
        self.max_token_observed = kwargs["max_length"]

        # optimized for f1-score
        self.binoculars_accuracy_threshold = kwargs["binoculars_accuracy_threshold"]

        # optimized for low-fpr [chosen at 0.01%]
        self.binoculars_fpr_threshold = kwargs["binoculars_fpr_threshold"]
        self.threshold = None
        self.observer_model = None
        self.performer_model = None

        self.change_mode(kwargs["mode"])  # low-fpr

        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name_or_path,
                                                       trust_remote_code=True)
        performer_tokenizer = AutoTokenizer.from_pretrained(self.performer_name_or_path,
                                                            trust_remote_code=True)

        self.assert_tokenizer_consistency(observer_tokenizer=self.tokenizer,
                                          performer_tokenizer=performer_tokenizer)

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            self.observer_name_or_path,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            self.performer_name_or_path,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.add_pad_token()

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = self.binoculars_fpr_threshold
        elif mode == "accuracy":
            self.threshold = self.binoculars_accuracy_threshold
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @staticmethod
    def assert_tokenizer_consistency(observer_tokenizer, performer_tokenizer):
        identical_tokenizers = (observer_tokenizer.vocab == performer_tokenizer.vocab)
        if not identical_tokenizers:
            raise ValueError(f"Tokenizers are not identical for {observer_tokenizer} and "
                             f"{performer_tokenizer}.")

    def _tokenize(self, batch: list[str]):
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings):
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        return observer_logits, performer_logits

    def compute_score(self, input_text):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits, performer_logits,
                        encodings, self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def inferencer_wrapper(self, sample):
        text1_score = self.compute_score(sample["text1"])
        text2_score = self.compute_score(sample["text2"])
        if text1_score < text2_score:
            return 0
        else:
            return 1

    def runner(self, samples):
        results = []
        for sample in tqdm.tqdm(samples):
            results.append(self.inferencer_wrapper(sample))
        return results


def runner_factory(samples: list, inference_methods: list, **kwargs):
    results = []
    method2inference_class = {"llama": LLamaInferencer,
                              "mistral": MistralInferencer,
                              "binocular": BinocularInferencer}

    for method in inference_methods:
        inference_obj = method2inference_class[method](**kwargs)
        results.append(inference_obj.runner(samples))
        del inference_obj

    return [calculate_mean(sample) for sample in zip(*results)]

