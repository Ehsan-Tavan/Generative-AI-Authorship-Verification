"""
    Generative_AI_Authorship_Verification Project:
        models:
            pooling_model.py

"""

from typing import List

import torch


class Pooling(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.pool_method2function = {
            "mean": self.mean_pool,
            "max": self.max_pool,
            "cls": self.cls_pool,
        }

    def get_embedding(self, sentence_embedding, attention_mask, pooling_methods: List[str]):
        output = []
        for method in pooling_methods:
            output.append(self.pool_method2function[method](sentence_embedding, attention_mask))
        return output

    @staticmethod
    def mean_pool(sentence_embedding, attention_mask):
        sentence_embedding = sentence_embedding * attention_mask.unsqueeze(-1)
        sentence_embedding = torch.sum(sentence_embedding, dim=1)
        num_non_padding_tokens = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1)
        sentence_embedding = sentence_embedding / num_non_padding_tokens
        return sentence_embedding

    @staticmethod
    def max_pool(sentence_embedding, attention_mask):
        sentence_embedding = sentence_embedding * attention_mask.unsqueeze(-1)
        # sentence_embedding[~attention_mask.unsqueeze(-1)] = float('-inf')
        sentence_embedding, _ = torch.max(sentence_embedding, dim=1)
        return sentence_embedding

    @staticmethod
    def cls_pool(sentence_embedding, _):
        return sentence_embedding[:, 0, :]

    def forward(self, sentence_embedding, attention_mask, pooling_methods: List[str]):
        return self.get_embedding(sentence_embedding, attention_mask, pooling_methods)
