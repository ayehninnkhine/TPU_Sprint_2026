
# 🌾 Fine-tuning Gemma 3 on Burmese Agriculture Dataset (TPU + Tunix + LoRA)

This project demonstrates how to fine-tune **Gemma 3 (4B)** on a **Burmese agriculture dataset** using:

* **Cloud TPU (v5 litepod-8)**
* **Tunix (JAX-based training)**
* 🔧 **LoRA / PEFT (efficient fine-tuning)**
* **BLEU evaluation (before vs after fine-tuning)**

The goal is to build a **domain-adapted Burmese language model** for agriculture-related Q&A and text generation.

## Dataset

We use the Burmese agriculture dataset from Hugging Face:

👉 [Burmese Agriculture Dataset](https://huggingface.co/datasets/AnkaNge/burmese_agriculture?)

* Language: Burmese 🇲🇲
* Domain: Agriculture
* Format: Instruction-style / QA-like

### Dataset Citation

If you use this dataset, please cite:

```bibtex
@dataset{ankanage_burmese_agriculture,
  author = {Yi Yi Chan Myae Win Shein},
  title = {Burmese Agriculture Dataset},
  year = {2024},
  url = {https://huggingface.co/datasets/AnkaNge/burmese_agriculture}
}
```

## Features

* TPU-based large model fine-tuning (Gemma 3)
* Efficient training using LoRA (low memory)
* Hugging Face dataset integration
* BLEU score comparison (base vs fine-tuned)
* Tunix-compatible training pipeline
* Support for **sharded checkpoints (Gemma 3 HF models)**


## Model

* **Base Model:** `google/gemma-3-4b-it`
* Architecture: Transformer (decoder-only LLM)
* Framework: JAX + Flax (via Tunix)



## TPU Setup (v5 litepod-8)

This project is designed for:

* **TPU type:** v5 litepod-8
* **Region:** e.g., `us-west4-a`
* **Devices:** 8 TPU cores

## ⚠️ Troubleshooting 

* If you encounter a `RESOURCE_EXHAUSTED` error during training or evaluation, reduce the batch size.

* Sharded Checkpoints

  Gemma 3 models (e.g., 4B, 12B) are stored in sharded safetensors format: 

  model.safetensors.index.json
  model-00001-of-00002.safetensors, etc.

  The original Tunix implementation assumes a single model.safetensors, which leads to file-not-found errors.

* Parameter Name Mismatch

  Tunix and Hugging Face use different parameter naming conventions:

  Tunix: layers.0.attn.k_einsum
  Hugging Face: language_model.model.layers.0.self_attn.k_proj.weight

  This causes errors such as: LoRA layer not found in base model state dict

* Please see my updated safetensors_saver.py at https://github.com/ayehninnkhine/tunix/blob/main/tunix/models/safetensors_saver.py . You can replace the corresponding file in your local Tunix installation if you encounter the same issue.

You can find the original file in your environment at: <your_venv>/lib/python3.12/site-packages/tunix/models/safetensors_saver.py


## Acknowledgement

Google Cloud credits are provided for this project. #TPUSprint 
