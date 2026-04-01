
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

## Acknowledgement

Google Cloud credits are provided for this project. #TPUSprint 
