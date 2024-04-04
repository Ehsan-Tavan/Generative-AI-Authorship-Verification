"""
    Generative_AI_Authorship_Verification Project:
        models:
            perplexity_classifier.py
"""
# ============================ Third Party libs =======================

import numpy as np
import torch
import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================ My packages ============================


class PerplexityClassifier:
    """
    Class for performing perplexity-based text classification using a pre-trained language model.

    Args:
        args: An object containing configuration arguments.

    Attributes:
        args: Configuration arguments.
        model: Pre-trained language model.
        tokenizer: Tokenizer for the language model.
    """

    def __init__(self, args):
        """
        Initializes the PerplexityClassifier object.

        Args:
            args: An object containing configuration arguments.
        """
        self.args = args
        self.model, self.tokenizer = None, None
        self.load_model()

    def load_model(self):
        """
        Loads the pre-trained language model and tokenizer.
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path,
                                                          device_map=self.args.device,
                                                          load_in_4bit=True,
                                                          do_sample=True,
                                                          torch_dtype="auto",
                                                          max_length=self.args.max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)

    def predict_perplexity(self, text: str):
        """
        Predicts the perplexity of the given text using the loaded language model.

        Args:
            text (str): Input text for which perplexity is to be predicted.

        Returns:
            float: Perplexity value of the input text.
        """
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        seq_len = tokenized_text.input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        max_length = self.model.config.max_length
        for begin_loc in range(0, seq_len, self.args.stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = tokenized_text.input_ids[:, begin_loc:end_loc].to(self.args.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.cpu().numpy()

    def evaluate(self, data: list):
        """
        Evaluates the perplexity-based classification model on the given data.

        Args:
            data (list): List of dictionaries containing text samples and their
            corresponding labels.

        Returns:
            tuple: A tuple containing the accuracy of the classification model as a float,
            and a list of predicted labels.
        """
        true_labels = [sample["label"] for sample in data]
        predicted_labels = []
        for sample in tqdm.tqdm(data):
            ppl1 = self.predict_perplexity(sample["text1"])
            ppl2 = self.predict_perplexity(sample["text2"])
            predicted_label = np.argmax([ppl1, ppl2])
            predicted_labels.append(predicted_label)
        accuracy = accuracy_score(true_labels, predicted_labels)
        return accuracy, predicted_labels
