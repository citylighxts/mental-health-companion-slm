"""
Sampling + generate narasi (chat format: label + narasi) dari dataset
ourafla/Mental-Health_Text-Classification_Dataset, mirip metodologi
proposal TA (sample per kategori -> LLM tulis narasi).

Dataset asli cuma punya `text` (postingan medsos) + `status` (label).
Script ini SAMPLING beberapa baris per label, terus panggil Claude buat
nulis narasi penjelasan gaya santai Gen Z, Bahasa Indonesia -- label-nya
sendiri dipakai langsung dari ground truth dataset (bukan ditentuin LLM),
biar konsisten & gak perlu validasi label terpisah kayak di TA.

Jalankan dulu dengan --limit kecil buat cek kualitas & estimasi biaya,
baru scale up.
"""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_CSV = SCRIPT_DIR / "../dataset/raw/mental_heath_unbanlanced.csv"
OUT_JSONL = SCRIPT_DIR / "../dataset/processed/narrative_dataset.jsonl"

SYSTEM_PROMPT = """Kamu adalah asisten kesehatan mental yang empatik, ngomong dengan gaya santai \
khas Gen Z Indonesia (boleh pakai kata sehari-hari, tapi tetap sopan dan gak toxic-positivity).

Kamu akan dikasih satu postingan media sosial (bahasa Inggris) dan kategori kondisi mental \
yang SUDAH ditentukan (jangan diubah, itu ground truth dari dataset). Narasi yang kamu tulis \
HARUS berpijak pada penanda linguistik yang tervalidasi jurnal berikut (rujuk salah satu yang \
relevan dengan isi postingan, jangan ngarang penanda sendiri):

- Suicidal: penggunaan kata ganti orang pertama tunggal ("aku"/"saya") berulang, negasi, dan \
  "social posturing" (nada seperti berpamitan/melepas identitas) -- (JMIR Formative Research \
  2022;6(8):e35563, DOI 10.2196/35563)
- Depression / Anxiety: sentimen bahasa yang dominan negatif dan berkorelasi dengan gejala \
  depresi/kecemasan -- (Stamatis et al., Depression and Anxiety 2022;39:794-804, \
  DOI 10.1002/da.23286)
- Normal: TIDAK menunjukkan penanda-penanda di atas secara dominan

Tugasmu nulis narasi singkat (2-4 kalimat) dalam Bahasa Indonesia yang:
1. Menyebutkan kategori kondisinya secara natural (bukan format kaku "kategori: X")
2. Menjelaskan kenapa postingan itu cocok masuk kategori tsb, secara EKSPLISIT ngerujuk penanda \
   linguistik dari daftar di atas yang muncul di teksnya (mis. "postingan ini banyak pakai kata \
   'aku', yang menurut riset [JMIR 2022] jadi salah satu penanda...")
3. Ngasih respons yang suportif/validating, sesuai kategori (kalau Suicidal, dorong cari bantuan \
profesional/hotline; kalau Depression/Anxiety, validasi perasaan + saran kecil; kalau Normal, \
respons santai biasa)

Jangan bikin diagnosis klinis formal, ini bukan alat medis -- posisikan sebagai teman yang \
paham & suportif, yang kebetulan ngerti riset. Jawab HANYA narasinya, tanpa embel-embel lain."""


def load_samples(n_per_class: int, seed: int = 0) -> list[dict]:
    with open(RAW_CSV) as f:
        rows = list(csv.DictReader(f))

    by_label = defaultdict(list)
    for row in rows:
        text = (row.get("text") or "").strip()
        label = (row.get("status") or "").strip()
        if text and label:
            by_label[label].append(row)

    rng = random.Random(seed)
    sampled = []
    for label, items in by_label.items():
        rng.shuffle(items)
        sampled.extend(items[:n_per_class])
    rng.shuffle(sampled)
    return sampled


def generate_narrative(client: anthropic.Anthropic, model: str, text: str, label: str) -> str:
    user_content = f'Postingan: "{text}"\nKategori (ground truth): {label}'
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return next(b.text for b in response.content if b.type == "text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=250, help="Jumlah sample per label")
    parser.add_argument("--limit", type=int, default=None, help="Batasi total baris (buat testing)")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model ID buat generate narasi")
    parser.add_argument("--dry-run", action="store_true", help="Cuma sampling, gak panggil API")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = load_samples(args.n_per_class, seed=args.seed)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Total sample yang akan diproses: {len(samples)}")
    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s["status"]] += 1
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] Gak manggil API. Contoh 2 baris pertama:")
        for s in samples[:2]:
            print(f"  text={s['text'][:60]!r}... label={s['status']}")
        return

    # Sebagian API key (identity-linked, biasanya dari akun yang terhubung ke
    # organisasi/SSO) butuh header anthropic-workspace-id eksplisit. Kalau
    # env var ANTHROPIC_WORKSPACE_ID diset, ikutkan sebagai default header.
    import os
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    extra_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else {}
    client = anthropic.Anthropic(default_headers=extra_headers)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, row in enumerate(samples, 1):
        text, label = row["text"], row["status"]
        for attempt in range(3):
            try:
                narasi = generate_narrative(client, args.model, text, label)
                break
            except anthropic.RateLimitError:
                wait = 5 * (attempt + 1)
                print(f"  [{i}] rate limited, retry in {wait}s...")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                print(f"  [{i}] SKIP (error {e.status_code}): {e.message}")
                narasi = None
                break
        else:
            narasi = None

        if narasi is None:
            continue

        results.append({
            "label": label,
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": narasi},
            ],
        })
        print(f"  [{i}/{len(samples)}] {label}: {narasi[:70]!r}...")

    with open(OUT_JSONL, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTersimpan {len(results)} baris ke {OUT_JSONL}")


if __name__ == "__main__":
    main()
