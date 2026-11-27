# ScamShield BD — NSUCEC NeuraHack 2026 Preliminary Round

## 1. Team Information
- **Team Name:** Shield
- **Members:** [FILL IN]
- **University:** [FILL IN]

## 2. Approach Summary
- **Preprocessing:** Unicode NFKC normalization; removal of `\u200b`, `\u200c`, and `\ufeff` while preserving `\u200d` (ZWJ) for Bangla; Cyrillic/Greek-to-Latin homoglyph mapping; Latin-only lowercasing; `[URL]` and `[NUM]` canonical tokens; `|||` conversation turn separator preservation.
- **Synthetic Data & Noise Augmentation (`synthetic_v3.csv`, 16,107 rows):**
  - Programmatically generated from scratch (`SEED = 42`) using compositional slot-filling across English, Bangla script, Banglish, and mixed text (~25% each), ~31%+ multi-turn dialogues (`|||`), real + invented brands, hard-negative contrast twins, 860 paraphrase variants of `sample.csv`, plus 1,107 Phase 3C targeted patch rows (Class 1 emergency/agent/KYC, short one-line Class 4 Banglish/Bangla, Class 0 hard-legitimate twins, Class 5 cashback promos, and 100 noise-only rows).
  - Adversarial noise (`apply_noise`) applied to ~30% of rows (typos, char repetition, spacing, random caps, emojis, leet-speak, Cyrillic homoglyphs, `\u200b`/`\u200c`, noise tokens, Bangla/Banglish variants, defanged URLs).
  - `test.csv` is never used for training, MLM, or pseudo-labeling.
- **Models & Ensemble:**
  - **Model 1 (2 seeds: `42` and `7`):** Three-head `xlm-roberta-base` (Task A 6-class softmax with class-weighted CrossEntropy, Task B 6-label sigmoid BCE, Task C 6-class softmax CrossEntropy; loss `0.5*L_A + 0.3*L_B + 0.2*L_C`), `fp16` autocast, `max_len=256` with head+tail truncation, `batch_size=32`, `3 epochs`, `lr=2e-5`, `10%` warmup. Trained on `synthetic_v3.csv` + `300` noise-only rows + `86` real clean & `86` real noisy rows (`5x` weight).
  - **Model 2:** `char_wb` `(2, 5)` + `word` `(1, 2)` TF-IDF + `LogisticRegression(class_weight="balanced")` trained on the exact same final set.
  - **Blend & Calibration:** `0.5 * (XLM-R_s42 + XLM-R_s7)` blended with TF-IDF using held-out-selected weights; Task B thresholds (`0.5` unless tuned+shrunk improves both Split 2 & Split 1 F0 by `> 0.01`); lowest-Brier Task A confidence calibrator fit on held-out probabilities, clipped to `[0.05, 0.98]`.
- **Post-Processing:** `label in {0, 5} -> techniques="NONE", action=0`; non-empty technique fallback for labels `1–4`; sorted `|`-joined codes; OOV noise-only rule (`label=0, techniques="NONE", action=0, confidence=0.5`).

## 3. External Models, Datasets, and Resources Used
- **Pretrained Model:** `xlm-roberta-base` from HuggingFace Hub (MIT License).
- **External Datasets:** None.
- **APIs / Other Resources:** None (AI coding assistant Antigravity was used to help write code).

## 4. Random Seeds
- `42` (primary seed for `random`, `numpy`, `sklearn`, `torch`, `transformers`, data generation, TF-IDF, and XLM-R seed 1) and `7` (XLM-R seed 2).

## 5. Environment and Hardware
- **Platform:** Kaggle Notebook with GPU (`NVIDIA Tesla T4` or `P100`, 16 GB VRAM) and Internet ON (for downloading `xlm-roberta-base`).
- **Software (Kaggle Python 3.10+ / CUDA 12.x):** `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch` (CUDA), `transformers`, `python-dotenv`.
- **Expected Runtime:** ~35–55 minutes total on Kaggle T4/P100 GPU (synthetic data generation < 15s, held-out Split 2 / Split 1 F0 ~10 min, final 2-seed XLM-R training ~15–20 min, 500k FP16 length-sorted inference ~12–18 min).

## 6. Step-by-Step Instructions to Run on Kaggle
1. Open a new Kaggle Notebook and attach the competition dataset (`nsucec-neurahack-2026-preliminary`, containing `sample.csv`, `test.csv`, `sample_submission.csv`).
2. In Notebook Settings, enable **GPU (T4 or P100)** and turn **Internet ON** (required to download `xlm-roberta-base` from HuggingFace).
3. Upload `notebook.ipynb` (and an empty `.env` if running locally).
4. Click **Run All**. The notebook recursively locates `sample.csv` and `test.csv` under `/kaggle/input`, regenerates `synthetic_v3.csv` from scratch, runs held-out blending/calibration, trains both final XLM-R seeds + TF-IDF, and runs 500k inference.

## 7. Expected Output
- `submission.csv` (`500,000` rows, columns `id,label,techniques,action,confidence`), validated inline before completion.

## 8. Known Limitations
- **FP16 GPU Non-Determinism:** CUDA `fp16` operations are not bit-for-bit deterministic across hardware/driver runs, so re-running the notebook can produce slightly different probabilities or boundary predictions.
- **Synthetic Data Bias:** While `synthetic_v3.csv` covers 100+ scenarios across 4 languages and hard-negative twins, synthetic distributions may still differ from unseen real-world test phrasing.
