#!/usr/bin/env python3
"""
TPU-based LoRA/QLoRA fine-tuning for Gemma on the Hugging Face `AnkaNge/burmese_agriculture` dataset

What this script does:
1. Logs into Hugging Face if HF_TOKEN is available
2. Downloads Gemma safetensors from Hugging Face
3. Loads Gemma into a JAX/TPU Tunix environment
4. Builds Grain dataloaders from the Burmese agriculture dataset
5. Fine-tunes with LoRA or QLoRA on TPU
6. Runs generation on the validation split for:
   - base model
   - fine-tuned LoRA model
7. Computes corpus BLEU with sacrebleu
8. Saves merged safetensors model and BLEU comparison JSON


"""

import os
import json
import gc
import shutil
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Iterable
import wandb
import dotenv
import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
import sacrebleu
from flax import nnx
from grain import python as grain
from huggingface_hub import snapshot_download

from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils
from tunix.sft.utils import show_hbm_usage


logger = logging.getLogger()
logger.setLevel(logging.INFO)


GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--dataset_name", type=str, default="AnkaNge/burmese_agriculture")
    parser.add_argument("--batch_size", type=int, default=16) # 64
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--eval_every_n_steps", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--rank", type=int, default=8) #16
    parser.add_argument("--alpha", type=float, default=None)

    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_examples", type=int, default=100)
    parser.add_argument("--max_generation_steps", type=int, default=96)

    parser.add_argument("--checkpoint_root", type=str, default="/tmp/content/gemma_4b")
    parser.add_argument("--output_dir", type=str, default="./gemma-burmese-agri-lora-4b")
    return parser.parse_args()


def create_dir(path: str):
    os.makedirs(path, exist_ok=True)
    logging.info("Created dir: %s", path)


def maybe_login_hf():
    dotenv.load_dotenv()
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.system(f'hf auth login --token "{hf_token}"')
        print("Logged into Hugging Face using HF_TOKEN.")
    else:
        print("HF_TOKEN not found. Skipping Hugging Face login.")


def get_model_config(model_id: str):
    if "gemma-3-270m" in model_id:
        return gemma3_model_lib.ModelConfig.gemma3_270m()
    if "gemma-3-4b" in model_id:
        return gemma3_model_lib.ModelConfig.gemma3_4b_it()
    raise ValueError(f"Unsupported model for this script: {model_id}")


def get_mesh():
    num_tpus = len(jax.devices())
    if num_tpus == 8:
        mesh_counts = (1,4)
    elif num_tpus == 1:
        mesh_counts = (1, 1)
    else:
        raise ValueError(f"Unsupported number of TPU devices: {num_tpus}")

    # Keep the original notebook behavior if needed.
    # You can change this if you want more advanced sharding later.
    # mesh_counts = (1, 1)
    return jax.make_mesh(
        mesh_counts,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )


def download_model(model_id: str):
    ignore_patterns = ["*.pth"]
    print(f"Downloading {model_id} from Hugging Face...")
    local_model_path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=ignore_patterns,
    )
    print(f"Model downloaded to: {local_model_path}")

    eos_tokens = []
    generation_config_path = os.path.join(local_model_path, "generation_config.json")
    if os.path.exists(generation_config_path):
        with open(generation_config_path, "r", encoding="utf-8") as f:
            generation_configs = json.load(f)
        eos_tokens = generation_configs.get("eos_token_id", [])
        print(f"Using EOS token IDs from generation config: {eos_tokens}")

    return local_model_path, eos_tokens


def load_base_model(local_model_path: str, model_id: str, mesh):
    model_config = get_model_config(model_id)
    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh
        )
    return base_model, model_config


def get_tokenizer(eos_tokens):
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
    print(f"Using EOS token IDs: {eos_tokens}")
    return tokenizer, eos_tokens


def build_prompt(question: str, crop: str) -> str:
    return (
        "user\n"
        "သင်သည် မြန်မာဘာသာစကားဖြင့် ဖြေကြားပေးသော စိုက်ပျိုးရေးအကူအညီပေးစနစ်ဖြစ်သည်။\n"
        "ပေးထားသောမေးခွန်း၏အဖြေကို မြန်မာဘာသာဖြင့် ပြန်လည်ဖြေဆိုပါ။ \n\n"
        f"မေးခွန်း: {question}\n\n"
        "model\n"
    )


class HFDatasetSource(grain.RandomAccessDataSource):
    def __init__(self, hf_dataset):
        self._ds = hf_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[int(idx)]


class _TokenizeAgriculture(grain.MapTransform):
    def __init__(self, tokenizer: tokenizer_lib.Tokenizer):
        self._tokenizer = tokenizer

    def map(self, element: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        prompt = build_prompt(
            question=element["question"],
            crop=element.get("crop", ""),
        )
        answer = element["answer"]

        src_tokens = self._tokenizer.tokenize(prompt, add_eos=False)
        dst_tokens = self._tokenizer.tokenize(answer, add_eos=True)
        return src_tokens, dst_tokens


class _BuildTrainInput(grain.MapTransform):
    def __init__(self, max_seq_len: int, pad_value: int | bool):
        self._max_seq_len = max_seq_len
        self._pad_value = pad_value

    def map(self, tokens: tuple[np.ndarray, np.ndarray]) -> peft_trainer.TrainingInput:
        src_tokens, dst_tokens = tokens

        tokens_all = np.concatenate([src_tokens, dst_tokens], axis=0)
        q_mask = np.zeros_like(src_tokens, dtype=np.bool_)
        a_mask = np.ones_like(dst_tokens, dtype=np.bool_)
        mask = np.concatenate([q_mask, a_mask], axis=0)

        if tokens_all.shape[0] > self._max_seq_len:
            tokens_all = tokens_all[: self._max_seq_len]
            mask = mask[: self._max_seq_len]

        tokens_all = self._pad_up_to_max_len(tokens_all, self._pad_value)
        mask = self._pad_up_to_max_len(mask.astype(np.int32), 0).astype(np.bool_)

        return peft_trainer.TrainingInput(
            input_tokens=tokens_all,
            input_mask=mask,
        )

    def _pad_up_to_max_len(self, input_tensor: np.ndarray, pad_value: int):
        seq_len = input_tensor.shape[0]
        to_pad = max(self._max_seq_len - seq_len, 0)
        return np.pad(
            input_tensor,
            [[0, to_pad]],
            mode="constant",
            constant_values=pad_value,
        )


class _FilterOverlength(grain.FilterTransform):
    def __init__(self, max_seq_len: int):
        self._max_seq_len = max_seq_len

    def filter(self, element: peft_trainer.TrainingInput) -> bool:
        return element.input_tokens.shape[0] <= self._max_seq_len


def build_data_loader(
    hf_dataset,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: tokenizer_lib.Tokenizer,
):
    data_source = HFDatasetSource(hf_dataset)
    return grain.DataLoader(
        data_source=data_source,
        sampler=grain.IndexSampler(
            num_records=len(data_source),
            num_epochs=num_epochs,
            shard_options=grain.NoSharding(),
        ),
        operations=[
            _TokenizeAgriculture(tokenizer),
            _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
            _FilterOverlength(max_seq_len),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
    )


def create_datasets(
    dataset_name: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: tokenizer_lib.Tokenizer,
    test_size: float,
    seed: int,
):
    import datasets

    ds = datasets.load_dataset(dataset_name, split="train")
    split = ds.train_test_split(test_size=test_size, seed=seed)

    train_ds = build_data_loader(
        split["train"],
        batch_size=global_batch_size,
        num_epochs=num_train_epochs,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
    )
    eval_ds = build_data_loader(
        split["test"],
        batch_size=global_batch_size,
        num_epochs=1,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
    )
    return train_ds, eval_ds, split["test"]


def gen_model_input_fn(x: peft_trainer.TrainingInput, tokenizer):
    pad_mask = x.input_tokens != tokenizer.pad_id()
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
        "input_tokens": x.input_tokens,
        "input_mask": x.input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
    }


def get_lora_model(base_model, rank: int, alpha: float, quantize: bool = False):
    if quantize:
        lora_provider = qwix.LoraProvider(
            module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
            rank=rank,
            alpha=alpha,
            weight_qtype="nf4",
            tile_size=128,
        )
    else:
        lora_provider = qwix.LoraProvider(
            module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
            rank=rank,
            alpha=alpha,
        )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model,
        lora_provider,
        **model_input,
    )
    return lora_model


def build_sampler(model, tokenizer, model_config):
    return sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )


def run_generation_eval(
    model,
    tokenizer,
    model_config,
    eval_hf_ds,
    eos_tokens,
    max_examples: int,
    max_generation_steps: int,
):
    sampler = build_sampler(model, tokenizer, model_config)

    n = min(max_examples, len(eval_hf_ds))
    examples = eval_hf_ds.select(range(n))

    prompts = []
    refs = []
    rows = []

    for ex in examples:
        prompt = build_prompt(ex["question"], ex.get("crop", ""))
        prompts.append(prompt)
        refs.append(ex["answer"])

    out_data = sampler(
        input_strings=prompts,
        max_generation_steps=max_generation_steps,
        eos_tokens=eos_tokens,
    )

    preds = []
    for i, pred in enumerate(out_data.text):
        pred = pred.strip()
        preds.append(pred)
        rows.append(
            {
                "index": i,
                "crop": examples[i].get("crop", ""),
                "question": examples[i]["question"],
                "reference": refs[i],
                "prediction": pred,
            }
        )

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu, rows


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    maybe_login_hf()

    if args.alpha is None:
        args.alpha = float(2 * args.rank)

    full_ckpt_dir = os.path.join(args.checkpoint_root, "full_ckpts")
    lora_ckpt_dir = os.path.join(args.checkpoint_root, "lora_ckpts")
    profiling_dir = os.path.join(args.checkpoint_root, "profiling")

    create_dir(full_ckpt_dir)
    create_dir(lora_ckpt_dir)
    create_dir(profiling_dir)
    create_dir(args.output_dir)

    print("\n--- HBM Usage BEFORE Model Load ---")
    show_hbm_usage()

    local_model_path, eos_tokens = download_model(args.model_id)

    mesh = get_mesh()
    base_model, model_config = load_base_model(local_model_path, args.model_id, mesh)
    tokenizer, eos_tokens = get_tokenizer(eos_tokens)

    print("Creating LoRA model...")
    lora_model = get_lora_model(
        base_model=base_model,
        rank=args.rank,
        alpha=args.alpha,
        quantize=args.use_quantization,
    )
    nnx.display(lora_model)
    print(f"Using {'QLoRA' if args.use_quantization else 'LoRA'} on TPU")

    print("Building agriculture dataset loaders...")
    train_ds, validation_ds, eval_hf_ds = create_datasets(
        dataset_name=args.dataset_name,
        global_batch_size=args.batch_size,
        max_target_length=args.max_target_length,
        num_train_epochs=args.num_epochs,
        tokenizer=tokenizer,
        test_size=args.test_size,
        seed=args.seed,
    )

    lora_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/tensorboard/lora",
        flush_every_n_steps=20,
    )

    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=args.eval_every_n_steps,
        max_steps=args.max_steps,
        metrics_logging_options=lora_logging_options,
        checkpoint_root_directory=lora_ckpt_dir,
    )

    trainer = peft_trainer.PeftTrainer(
        lora_model,
        optax.adamw(args.learning_rate),
        training_config,
    ).with_gen_model_input_fn(lambda x: gen_model_input_fn(x, tokenizer))

    print("Starting TPU fine-tuning...")
    trainer.train(train_ds, validation_ds)

    print("\nEvaluating base model BLEU...")
    wandb.init()
    base_bleu, base_rows = run_generation_eval(
        model=base_model,
        tokenizer=tokenizer,
        model_config=model_config,
        eval_hf_ds=eval_hf_ds,
        eos_tokens=eos_tokens,
        max_examples=args.max_eval_examples,
        max_generation_steps=args.max_generation_steps,
    )

    print("\nEvaluating fine-tuned LoRA model BLEU...")
    lora_bleu, lora_rows = run_generation_eval(
        model=lora_model,
        tokenizer=tokenizer,
        model_config=model_config,
        eval_hf_ds=eval_hf_ds,
        eos_tokens=eos_tokens,
        max_examples=args.max_eval_examples,
        max_generation_steps=args.max_generation_steps,
    )

    bleu_summary = {
        "dataset_name": args.dataset_name,
        "model_id": args.model_id,
        "eval_examples": min(args.max_eval_examples, len(eval_hf_ds)),
        "base_bleu": {
            "score": base_bleu.score,
            "bp": base_bleu.bp,
            "sys_len": base_bleu.sys_len,
            "ref_len": base_bleu.ref_len,
            "precisions": base_bleu.precisions,
        },
        "lora_bleu": {
            "score": lora_bleu.score,
            "bp": lora_bleu.bp,
            "sys_len": lora_bleu.sys_len,
            "ref_len": lora_bleu.ref_len,
            "precisions": lora_bleu.precisions,
        },
        "bleu_improvement": lora_bleu.score - base_bleu.score,
    }

    save_json(os.path.join(args.output_dir, "bleu_comparison.json"), bleu_summary)
    save_jsonl(os.path.join(args.output_dir, "base_predictions.jsonl"), base_rows)
    save_jsonl(os.path.join(args.output_dir, "lora_predictions.jsonl"), lora_rows)

    print("\nSaving merged LoRA model...")
    merged_dir = os.path.join(args.output_dir, "merged_model")
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    os.makedirs(merged_dir, exist_ok=True)

    gemma_params.save_lora_merged_model_as_safetensors(
        local_model_path=local_model_path,
        output_dir=merged_dir,
        lora_model=lora_model,
        rank=args.rank,
        alpha=args.alpha,
    )

    print("\n=== Done ===")
    print(json.dumps(bleu_summary, ensure_ascii=False, indent=2))
    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
