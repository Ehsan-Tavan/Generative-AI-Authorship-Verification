# ============================ Third Party libs ============================
import os
from tqdm import tqdm
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, \
    pipeline
from trl import AutoModelForCausalLMWithValueHead, create_reference_model, PPOTrainer, PPOConfig
from trl.core import LengthSampler

# ============================ My packages ============================
from src.configuration import BaseConfig
from src.data_loader import load_jsonl
from src.data_preparation import paraphraser_data_creator


def collator(data):
    samples = dict((key, [d[key] for d in data]) for key in data[0])
    return samples


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: " \
           f"{all_model_params}\npercentage of trainable model parameters: " \
           f"{100 * trainable_model_params / all_model_params:.2f}%"


def calculate_reward(predicted_sample, toxicity_tokenizer, toxicity_model):
    toxicity_input_ids = toxicity_tokenizer(predicted_sample, return_tensors="pt").input_ids

    logits = toxicity_model(input_ids=toxicity_input_ids).logits
    print(f'logits [not hate, hate]: {logits.tolist()[0]}')

    # Print the probabilities for [not hate, hate]
    probabilities = logits.softmax(dim=-1).tolist()[0]
    print(f'probabilities [not hate, hate]: {probabilities}')

    # get the logits for "not hate" - this is the reward!
    not_hate_index = 0
    toxicity_reward = (logits[:, not_hate_index]).tolist()
    print(f'reward (high): {toxicity_reward}')
    return toxicity_reward


def build_dataset(tokenizer,
                  dataset):
    def tokenize(sample):
        # Wrap each dialogue with the instruction.
        sample["input_ids"] = tokenizer.encode(sample["instruction"])
        # This must be called "query", which is a requirement of our PPO library.
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # Tokenize each dialogue.
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    return dataset


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TRAIN_DATA = load_jsonl(os.path.join(ARGS.processed_data_dir, ARGS.pair_train_file))[:64]
    print(f"We have {len(TRAIN_DATA)} training samples.")
    TRAIN_SAMPLES = paraphraser_data_creator(TRAIN_DATA)
    TRAIN_DATASET = Dataset.from_list(TRAIN_SAMPLES)

    base_model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_path,  # Llama 2 7B, same as before
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        use_flash_attention_2=False
    )

    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_path, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # lora_config = LoraConfig(r=32,  # Rank
    #                          lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05,
    #                          bias="none",
    #                          task_type="CAUSAL_LM")

    peft_model = PeftModel.from_pretrained(base_model,
                                           "/mnt/disk2/ehsan.tavan/gen_ai/assets/saved_model/"
                                           "Paraphraser/version_1/checkpoint-1",
                                           is_trainable=True,
                                           torch_dtype=torch.bfloat16)
    # lora_config=lora_config)

    print(f"PEFT model parameters to be updated:\n"
          f"{print_number_of_trainable_model_parameters(peft_model)}\n")

    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model,
                                                                  torch_dtype=torch.bfloat16,
                                                                  is_trainable=True)

    print(
        f'PPO model parameters to be updated (ValueHead + 769 params):\n'
        f'{print_number_of_trainable_model_parameters(ppo_model)}\n')
    print(ppo_model.v_head)

    ref_model = create_reference_model(ppo_model)
    # device = 0 if torch.cuda.is_available() else "cpu"

    print(f'Reference model parameters to be updated:\n'
          f'{print_number_of_trainable_model_parameters(ref_model)}\n')

    reward_tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/LanguageModels/xlm-roberta-base")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "/mnt/disk2/ehsan.tavan/gen_ai/assets/saved_model/Generative_AI_Authorship_Verification/"
        "version_0/best_model")

    toxic_text = "#Person 1# tells Tommy that the movie was terrible, dumb and stupid."

    print(calculate_reward(toxic_text, reward_tokenizer, reward_model))

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="/mnt/disk2/ehsan.tavan/gen_ai/assets/saved_model/"
              "Generative_AI_Authorship_Verification/version_0/best_model",
        device=ARGS.device)

    reward_logits_kwargs = {
        "top_k": None,  # Return all scores.
        "function_to_apply": "none",  # Set to "none" to retrieve raw logits.
        "batch_size": 16
    }

    reward_probabilities_kwargs = {
        "top_k": None,  # Return all scores.
        "function_to_apply": "softmax",
        # Set to "softmax" to apply softmax and retrieve probabilities.
        "batch_size": 16
    }
    print("Reward model output:")
    print("For non-toxic text")
    print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
    print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))

    learning_rate = 1.41e-5
    max_ppo_epochs = 1
    mini_batch_size = 4
    batch_size = 16

    config = PPOConfig(
        model_name=ARGS.model_path,
        learning_rate=learning_rate,
        ppo_epochs=max_ppo_epochs,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size
    )

    TRAIN_DATASET = build_dataset(dataset=TRAIN_DATASET,
                                  tokenizer=tokenizer)

    ppo_trainer = PPOTrainer(config=config,
                             model=ppo_model,
                             ref_model=ref_model,
                             tokenizer=tokenizer,
                             dataset=TRAIN_DATASET,
                             data_collator=collator)

    """
       The fine-tuning loop consists of the following main steps:

       - Get the query responses from the policy LLM (PEFT model).
       - Get sentiments for query/responses from hate speech RoBERTa model.
       - Optimize policy with PPO using the (query, response, reward) triplet.

       The operation is running if you see the following metrics appearing:

       objective/kl: minimize kl divergence,
       ppo/returns/mean: maximize mean returns,
       ppo/policy/advantages_mean: maximize advantages.
       """

    not_hate_index = 0
    output_min_length = 300
    output_max_length = 800
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "min_length": 5,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True
    }

    reward_kwargs = {
        "top_k": None,  # Return all scores.
        "function_to_apply": "none",  # You want the raw logits without softmax.
        "batch_size": 16
    }

    max_ppo_steps = 10

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break

        prompt_tensors = batch["input_ids"]

        # Get response from FLAN-T5/PEFT LLM.
        summary_tensors = []

        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()

            generation_kwargs["max_new_tokens"] = max_new_tokens
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

            summary_tensors.append(summary.squeeze()[-max_new_tokens:])

        # This needs to be called "response".
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # Compute reward outputs.
        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

        # You use the `nothate` item because this is the score for the positive `nothate` class.
        reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]

        # Run PPO step.
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-'.join('' for x in range(100)))
