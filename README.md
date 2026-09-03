# genz-mentalhealth-slm

Model chat kesehatan mental 4-kelas (**Suicidal**, **Depression**, **Anxiety**, **Normal**),
di-fine-tune (LoRA) di atas `Llama-3.2-3B-Instruct`, ditujukan buat pengguna Gen Z yang
capek/burnout — respons dengan nada santai, empatik, dan Bahasa Indonesia.

<img width="1501" height="362" alt="Screenshot 2026-09-03 at 17 22 30" src="https://github.com/user-attachments/assets/b6056bb8-e456-4f4b-9e6a-29300f58862e" />

## Pendekatan

- **Framing**: teman curhat (companion), bukan alat klinis. 4 label
  (`Suicidal`/`Depression`/`Anxiety`/`Normal`) dari `ourafla/Mental-Health_Text-Classification_Dataset`
  dipakai hanya sebagai steering internal saat generate (aturan krisis vs non-krisis)
  dan metadata buat stratified split — model tidak pernah menyebut label ke user.
- **Dataset**: `scripts/generate_dataset.py` — tiap baris CSV di-transcreate Claude jadi
  pesan chat orang pertama gaya Gen Z Indonesia (slang, typo, singkatan), lalu dilanjut
  jadi percakapan 1–6 giliran. ~70% multi-turn. Tiap balasan asisten wajib lolos
  validator di `scripts/dataset_validators.py` (nol tanda tanya, 1–4 kalimat, mayoritas
  Bahasa Indonesia, aturan hotline krisis) — yang gagal di-regenerate lalu di-drop.
- **Fine-tuning**: LoRA (rank 8, 16 layer terakhir) pakai [mlx-lm](https://github.com/ml-explore/mlx-lm)
  di Apple Silicon, `mask_prompt=true`.
- **Target deployment**: on-device iOS (Core ML, kuantisasi INT4).

## Struktur

```
scripts/
  generate_dataset.py     # sampling + generate percakapan pakai Claude (+ validator)
  dataset_validators.py   # cek struktur & gaya tiap percakapan (unit-tested)
  companion_prompt.py     # system prompt + few-shot buat generate
  split_dataset.py        # stratified split train/valid/test (label + turn-type)
  merge_lora_to_hf.py     # merge adapter LoRA (mlx) -> checkpoint HF PyTorch
  load_model.py           # utilitas load model
dataset/processed/        # conversations.jsonl + train/valid/test.jsonl
training/
  lora_config.yaml        # config LoRA (mlx-lm)
  adapters/                # (gitignored — hasil training, regenerate sendiri)
docs/references.md         # sitasi jurnal buat grounding narasi tiap label
```

## Catatan

- Untuk kasus krisis (label `Suicidal`), respons selalu mengarahkan ke hotline
  **Into The Light Indonesia — 119 ext 8**.
- Model & checkpoint (`models/`, `training/adapters/`) sengaja tidak disertakan di repo
  ini (ukuran GB) — regenerate lewat `training/lora_config.yaml` + `mlx_lm.lora`.
