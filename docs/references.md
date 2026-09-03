# Referensi — Grounding Narasi Kesehatan Mental

Dipakai sebagai acuan definisi tingkat keparahan tiap kategori saat sampling seed di
`scripts/generate_dataset.py`. Penanda linguistik di bawah **tidak** muncul di dalam
balasan chatbot — model diposisikan sebagai teman curhat, bukan alat klasifikasi. Sitasi
ini murni dokumentasi kenapa 4 kategori itu dipisah begitu.

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
  **bukan** kriteria diagnosis klinis formal (DSM-5 penuh) — respons yang digenerate
  diarahkan untuk **tidak** mengklaim diagnosis medis (lihat `SYSTEM_PROMPT` di
  `scripts/companion_prompt.py`).
- Label `status` dari dataset sumber dipakai apa adanya sebagai steering internal generate
  (krisis vs non-krisis) + kunci stratifikasi split. Balasan yang digenerate adalah respons
  companion — tidak menyebut kategori, tidak mengklaim diagnosis, tidak mengutip riset.
