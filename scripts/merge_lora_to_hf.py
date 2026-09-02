"""
Merge the mlx-lm trained LoRA adapter (training/adapters/adapters.safetensors)
into the HuggingFace PyTorch base model (models/hf-base-model), producing a
fully-fused HF-format checkpoint at models/fused-hf-model.

Merge formula (matches mlx_lm.tuner.lora.LoRALinear.fuse exactly):
    W_new = W_original + scale * (lora_b.T @ lora_a.T)

Key naming in the adapter file mirrors HF's LlamaForCausalLM naming 1:1:
    model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_a / .lora_b
    model.layers.{i}.mlp.{gate,up,down}_proj.lora_a / .lora_b
So the base weight key is the same prefix with `.weight` instead of `.lora_a/.lora_b`.
"""
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_DIR = "models/hf-base-model"
ADAPTER_PATH = "training/adapters/adapters.safetensors"
ADAPTER_CONFIG_PATH = "training/adapters/adapter_config.json"
OUT_DIR = "models/fused-hf-model"


def main():
    with open(ADAPTER_CONFIG_PATH) as f:
        adapter_config = json.load(f)
    scale = adapter_config["lora_parameters"]["scale"]
    print(f"LoRA scale: {scale}")

    print("Loading LoRA adapter weights...")
    lora_weights = {}
    with safe_open(ADAPTER_PATH, framework="pt") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key)

    # Group lora_a/lora_b pairs by their target weight key
    targets = {}
    for key, tensor in lora_weights.items():
        if key.endswith(".lora_a"):
            base_key = key[: -len(".lora_a")] + ".weight"
            targets.setdefault(base_key, {})["a"] = tensor
        elif key.endswith(".lora_b"):
            base_key = key[: -len(".lora_b")] + ".weight"
            targets.setdefault(base_key, {})["b"] = tensor
    print(f"Found {len(targets)} target weight matrices to merge.")

    print("Loading base HF model (this can take a bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, torch_dtype=torch.bfloat16
    )
    state_dict = model.state_dict()

    merged_count = 0
    for base_key, pair in targets.items():
        if base_key not in state_dict:
            raise KeyError(f"Base model has no weight named: {base_key}")
        lora_a = pair["a"].to(torch.float32)  # [in, r]
        lora_b = pair["b"].to(torch.float32)  # [r, out]
        delta = (scale * lora_b.T) @ lora_a.T  # [out, in]
        w = state_dict[base_key].to(torch.float32)
        if w.shape != delta.shape:
            raise ValueError(
                f"Shape mismatch for {base_key}: base {w.shape} vs delta {delta.shape}"
            )
        state_dict[base_key] = (w + delta).to(torch.bfloat16)
        merged_count += 1

    print(f"Merged {merged_count} weight matrices.")
    model.load_state_dict(state_dict)

    print(f"Saving fused HF model to {OUT_DIR} ...")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)

    print("Copying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("Done. Fused HF model saved at:", OUT_DIR)


if __name__ == "__main__":
    main()
