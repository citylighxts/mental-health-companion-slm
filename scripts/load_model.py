import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_choice):
    # Mapping nama model ke ID Hugging Face
    model_mapping = {
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "phi": "microsoft/Phi-3-mini-4k-instruct"
    }

    if model_choice not in model_mapping:
        raise ValueError(f"Model {model_choice} tidak didukung. Pilih: llama, qwen, phi")

    model_id = model_mapping[model_choice]
    print(f"[*] Memuat tokenizer untuk {model_id}...")
    
    # 1. Memuat Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Llama 3 dan Qwen biasanya butuh pad_token jika kita mau batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Memuat model {model_id} ke Mac M5 (MPS)...")
    
    # 2. Memuat Model
    # Mac M5 menggunakan chip Apple Silicon, jadi kita arahkan komputasi ke "mps" (Metal Performance Shaders)
    # Kita muat dalam presisi float16 atau bfloat16 untuk menghemat memori (Mac M-series sangat efisien dengan memori terpadu)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Gunakan bfloat16 atau float16
        device_map="mps",           # Wajib "mps" untuk akselerasi GPU Mac M5
    )

    print("[*] Sukses! Model dan Tokenizer berhasil dimuat.")
    print(f"    - Perangkat yang digunakan: {model.device}")
    print(f"    - Presisi model: {model.dtype}")
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load LLM Model and Tokenizer for Mac M5")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "phi"], default="qwen", help="Pilih model untuk dimuat")
    
    args = parser.parse_args()
    
    # Panggil fungsi load
    model, tokenizer = load_model_and_tokenizer(args.model)
