"""
    Generative_AI_Authorship_Verification Project:
        models:
            llmModel.py

"""

# ============================ Third Party libs =======================
import os
from typing import List

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig, \
    TrainingArguments, DataCollatorWithPadding, Trainer, LlamaForSequenceClassification, \
    AutoModelForSequenceClassification

# ============================ My packages ============================
from src.utils import CreateLogFile


class LLMModel:
    def __init__(self, model_path: str, label2id: dict, id2label: dict, args):
        self.label2id = label2id
        self.id2label = id2label
        self.model_path = model_path
        self.args = args

        self.model, self.tokenizer = self.load_model(self.model_path)

        self.create_log_file = CreateLogFile(os.path.join(self.args.saved_model_path,
                                                          self.args.model_name))

        self.peft_model = None
        self.peft_config = None
        self.eval_dataset = None

    def create_quantization_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=self.args.load_in_8bit,  # load model in 8-bit precision
            load_in_4bit=self.args.load_in_4bit,  # load model in 4-bit precision
            bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
            bnb_4bit_use_double_quant=self.args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # During computation, pre-trained model should be loaded in BF16 format
        )
        return bnb_config

    def create_lora_config(self, modules):
        self.peft_config = LoraConfig(
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            r=self.args.lora_rank,
            bias="none",  # setting to 'none' for only training weight params instead of biases
            task_type=TaskType.SEQ_CLS,
            target_modules=modules
        )

    def load_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Needed for LLaMA tokenizer
        # tokenizer.pad_token = tokenizer.eos_token

        bnb_config = self.create_quantization_config()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            device_map=self.args.device
        )
        model.config.use_cache = False

        # DM added
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        try:
            model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
        except Exception as error:
            print(error)
            print("Warning: Exception occurred while setting pad_token_id")
        return model, tokenizer

    def find_all_linear_names(self) -> List[str]:
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    def create_peft_model(self):
        lora_module_names = self.find_all_linear_names()
        self.create_lora_config(modules=lora_module_names)

        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
        self.peft_model = get_peft_model(self.model, self.peft_config)

    def fine_tune(self, train_dataset, eval_dataset):
        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True,
                                                    fn_kwargs={"tokenizer": self.tokenizer,
                                                               "max_length": self.args.max_length})

        tokenized_valid_dataset = eval_dataset.map(preprocess_function, batched=True,
                                                   fn_kwargs={"tokenizer": self.tokenizer,
                                                              "max_length": self.args.max_length})
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        saved_model_path = self.create_log_file.create_versioned_file()
        num_evaluate_steps = int(len(train_dataset) / (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps))
        training_arguments = TrainingArguments(
            output_dir=saved_model_path,
            auto_find_batch_size=True,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            optim=self.args.optim,
            num_train_epochs=self.args.num_train_epochs,
            evaluation_strategy=self.args.evaluation_strategy,
            save_strategy=self.args.save_strategy,
            learning_rate=self.args.learning_rate,
            bf16=False,
            save_steps=num_evaluate_steps,
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=self.args.warmup_ratio,
            group_by_length=True,
            logging_first_step=True,
            logging_steps=num_evaluate_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            tf32=False,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            save_total_limit=2,  # will save 2 checkpoints (best one and last one)
        )
        if self.peft_model:
            trainer = CustomTrainer(
                model=self.peft_model,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_valid_dataset,
                tokenizer=self.tokenizer,
                args=training_arguments,
                data_collator=data_collator,
            )

            for name, module in trainer.model.named_modules():
                if "norm" in name:
                    module = module.to(torch.float32)
            self.peft_model.config.use_cache = False
            trainer.train()
            trainer.save_model(os.path.join(saved_model_path, "best_model"))
        else:
            raise Exception("Peft model is not created!!!")


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits[:, 1], labels.to(
            torch.float32))  # , pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True,
                                  padding=True,
                                  max_length=fn_kwargs["max_length"])
