# Referensi — Grounding Narasi Kesehatan Mental

Dipakai sebagai "ground truth" acuan pas Claude Sonnet 5 (mother LLM) generate narasi di
`scripts/generate_narrative.py`. Pendekatan ini niru pola Tabel 3.1 di proposal TA (threshold
yang disitasi dari jurnal terverifikasi), diterapkan ke domain kesehatan mental berbasis teks.

## Suicidal

**Kim, D., et al.** (2022). Analyzing Suicide Risk From Linguistic Features in Social Media:
Evaluation Study. *JMIR Formative Research*, 6(8), e35563.
DOI: [10.2196/35563](https://doi.org/10.2196/35563)

Penanda linguistik yang diidentifikasi:
- Penggunaan kata ganti orang pertama tunggal ("aku"/"saya"/"I") berulang
- Negasi
- *Social posturing* — nada linguistik terkait autentisitas & *clout* (LIWC), sering muncul
  sebagai nada "berpamitan"/melepas identitas

## Depression & Anxiety

**Stamatis, C. A., Meyerhoff, J., Liu, T., Sherman, G., Wang, H., Liu, T., Curtis, B., Ungar,
L. H., & Mohr, D. C.** (2022). Prospective associations of text-message-based sentiment with
symptoms of depression, generalized anxiety, and social anxiety. *Depression and Anxiety*,
39, 794–804.
DOI: [10.1002/da.23286](https://doi.org/10.1002/da.23286)

Temuan: sentimen bahasa (dari pesan teks sehari-hari) yang dominan negatif berkorelasi
prospektif dengan gejala depresi, generalized anxiety disorder, dan social anxiety.

## Normal

Definisional — postingan yang tidak menunjukkan penanda-penanda di atas secara dominan.
Tidak memerlukan sitasi terpisah (baseline kategori, bukan klaim positif dari studi).

## Catatan batasan

- Dua sumber di atas fokus ke *linguistic markers* yang berkorelasi dengan kondisi mental,
  **bukan** kriteria diagnosis klinis formal (DSM-5 penuh) — narasi hasil generate eksplisit
  diarahkan untuk **tidak** mengklaim diagnosis medis (lihat `SYSTEM_PROMPT` di
  `generate_narrative.py`).
- Dataset sumber (`ourafla/Mental-Health_Text-Classification_Dataset`) sudah punya label
  ground truth sendiri (`status`) hasil kurasi dari 3 corpus publik — narasi yang digenerate
  di sini **menjelaskan** label yang sudah ada, bukan menentukan ulang labelnya.
