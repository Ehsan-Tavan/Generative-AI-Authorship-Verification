# Generative-AI-Authorship-Verification

Welcome to the Generative-AI-Authorship-Verification Framework. This README will guide you through the process of setting up the required models, pulling additional data, and running the inference framework.

**Requirements**

To run this framework, you need to download the following language models:

- [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [Falcon-7b-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

Additionally, download the following custom LoRA weights for the models:

- [Generative-AV-Mistral-v0.1-7b](https://huggingface.co/Ehsan-Tavan/Generative-AV-Mistral-v0.1-7b)
- [Generative-AV-LLaMA-2-7b](https://huggingface.co/Ehsan-Tavan/Generative-AV-LLaMA-2-7b)

**Setting Up**

Before running the framework, ensure all models and LoRA weights are downloaded to accessible locations on your system.
Running the Framework

To run the framework, execute inferencer.py with the required arguments. Here's an example command structure:

```python
python inferencer.py \
  --llama_model_path="Path to Llama-2-7b" \
  --llama_peft_model_path="Path to Generative-AV-LLaMA-2-7b" \
  --mistral_model_path="Path to Mistral-7B-v0.1" \
  --mistral_peft_model_path="Path to Generative-AV-Mistral-v0.1-7b" \
  --observer_name_or_path="Path to Falcon-7b" \
  --performer_name_or_path="Path to Falcon-7b-Instruct" \
  --outputDir="Path to output .jsonl file" \
  --inputDataset="Path to input .jsonl file"
```

Replace the text within double quotes with the correct paths to your downloaded models and files.

**Output**

The framework will output a .jsonl file to the location specified in outputDir. This file contains the results of the inference process, which you can further process or analyze as needed.

**Push to TIRA**

You can push this software to tira via:

```
tira-run \
    --image pan24-generative-authorship:latest \
    --input-dataset generative-ai-authorship-verification-panclef-2024/pan24-generative-authorship-tiny-smoke-20240417-training \
    --mount-hf-model meta-llama/Llama-2-7b-hf Ehsan-Tavan/Generative-AV-LLaMA-2-7b mistralai/Mistral-7B-v0.1 Ehsan-Tavan/Generative-AV-Mistral-v0.1-7b tiiuae/falcon-7b tiiuae/falcon-7b-instruct \
    --command 'python /app/src/inferencer.py --llama_model_path=meta-llama/Llama-2-7b-hf --llama_peft_model_path=/root/.cache/huggingface/hub/models--Ehsan-Tavan--Generative-AV-LLaMA-2-7b/snapshots/3df014e07f262611b4eb9daa8a75cc486702b138/ --mistral_model_path=mistralai/Mistral-7B-v0.1 --mistral_peft_model_path=/root/.cache/huggingface/hub/models--Ehsan-Tavan--Generative-AV-Mistral-v0.1-7b/snapshots/593a0ade0090e3988824f7c05779360d24c5048e/ --observer_name_or_path=/root/.cache/huggingface/hub/models--tiiuae--falcon-7b/snapshots/898df1396f35e447d5fe44e0a3ccaaaa69f30d36/ --performer_name_or_path=/root/.cache/huggingface/hub/models--tiiuae--falcon-7b-instruct/snapshots/cf4b3c42ce2fdfe24f753f0f0d179202fea59c99/ --outputDir=$outputDir/results.jsonl --inputDataset=$inputDataset/dataset.jsonl' \
    --push true
```
