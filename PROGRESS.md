# ScamShield BD — Project Progress & Team/Agent Handoff

**Competition:** NSUCEC NeuraHack 2026 Preliminary Round (*ScamShield BD: Advanced Multilingual Conversation Intelligence*)  
**Team Name:** `Shield`  
**Members:** Asraful Islam Afridi, Linkon Biswas, Anninda Raj Das  
**University:** Gopalganj Science and Technology University  
**Deadlines (BST, UTC+6):**
- **Kaggle Submission Deadline:** 27 November 2026, 6:00 PM BST
- **Final ZIP Package (`NeuraHack_Preli_Shield.zip`) Deadline:** 27 November 2026, 10:00 PM BST

---

## 1. Strict Competition Rules & Engineering Constraints
1. **Official Problem Statement overrides any PRD conflict.**
2. **No `test.csv` training/leakage:** Never copy, paraphrase, pseudo-label, or run MLM on `test.csv`. `test.csv` is used **only** for final inference (and the 10k-row throughput benchmark).
3. **Preprocessing rule:** Strip `\u200b`, `\u200c`, `\ufeff` **only**; **keep `\u200d` (ZWJ)** because Bangla conjuncts depend on it. Do not use consonant-run noise regexes.
4. **No hardcoded action-priority or class-priority rules:** Learn Task B (`T1–T6`) and Task C (`0–5`) from data.
5. **Post-processing rule:** Enforce only `label == 0 -> techniques = "NONE", action = 0`. If any non-zero label predicts an empty technique set, fall back to the single highest-probability technique. Sort technique codes and join with `|`.
6. **Reproducibility:** Fixed `SEED = 42` everywhere (`random`, `numpy`, `sklearn`, `torch`, `transformers`). No hardcoded API keys (`.env` only; currently empty). Every external dataset, pretrained model, and API must be declared in `notebook.ipynb` and `README.md`.

---

## 2. Current Workspace Inventory

### Root Directory
- `sample.csv` — 86 labeled seed rows (`id`, `text`, `label`, `techniques`, `action`), 0 nulls
- `test.csv` — 500,000 unlabeled test rows (`id`, `text`), 0 nulls, 21.45% multi-turn (`|||`), 10.97% Bangla script
- `sample_submission.csv` — 500,000-row submission template (`id`, `label`, `techniques`, `action`, `confidence`)
- `submission.csv` — Phase 1 TF-IDF baseline predictions (500,000 rows, validated `PASS`)
- `submission_phase2.csv` — Phase 2 TF-IDF + `synthetic.csv` predictions (500,000 rows, validated `PASS`)
- `submission_phase3a.csv` — Phase 3A TF-IDF + `synthetic_v2.csv` predictions (500,000 rows, validated `PASS`)
- `notebook.ipynb` — Local self-contained reproducible notebook (currently synced to **Phase 3A**, executed top-to-bottom, SHA-256 verified against `submission_phase3a.csv`)
- `kaggle_phase3.ipynb` — **Ready-to-upload Kaggle GPU notebook** for Phase 3B (TF-IDF vs Frozen E5-base vs Frozen MiniLM vs Fine-tuned 3-head `xlm-roberta-base` + 10k inference benchmark). *Not yet run on Kaggle.*
- `NeuraHack_Preli_Shield.zip` — Draft Phase 1 ZIP (do **not** submit as final; rebuild at the end with `scripts/build_package.py` once the final model/submission is chosen)
- `README.md`, `requirements.txt`, `.env` (empty, no API keys used)

### `src/`
- `src/inspect_data.py` — Step 1 data inspection (`sample.csv`, `test.csv`, `sample_submission.csv`)
- `src/preprocess.py` — NFKC normalization, `\u200b`/`\u200c`/`\ufeff` removal (keeping `\u200d`), Cyrillic/Greek homoglyph mapping, Latin-only lowercasing, `[URL]` & `[NUM]` tokens, `|||` preservation, plus unit tests
- `src/validate_submission.py` — Strict validator (`PASS`/`FAIL`) checking headers, 500k IDs/order, ranges, technique formatting, confidence `[0, 1]`, and no nulls/blanks
- `src/baseline.py` — Phase 1 TF-IDF (`char_wb` 2–5 + `word` 1–2) + `LogisticRegression(class_weight="balanced")`
- `src/noise.py` — `apply_noise(text, rng, level)` for English, Banglish, and Bangla script (typos, char repeats, space drops/adds, random caps, emojis, leet-speak, Cyrillic homoglyphs, `\u200b`/`\u200c`, noise strings, Bangla/Banglish variants, defanged URLs)
- `src/synth.py` — Phase 2 compositional generator (6,000 rows)
- `src/phase2_pipeline.py` — Phase 2 honest validation `(a)-(d)` + `submission_phase2.csv`
- `src/phase25_diagnostic.py` — Phase 2.5 cosine leakage check & scenario-held-out `StratifiedGroupKFold`
- `src/synth_v2.py` — **Phase 3A expanded generator** (15,000 rows, 15–21 scenarios/class, hard-negative twins, 860 `real_paraphrase` rows)
- `src/phase3a_eval.py` — Phase 3A scenario-held-out `StratifiedGroupKFold` evaluation + 40-row quality check
- `src/phase3a_check.py` — Phase 3A real-row check (1A & 1B), cross-scenario cosine check, and `submission_phase3a.csv` generation

### `scripts/`
- `scripts/create_and_run_notebook.py` — Builds self-contained `notebook.ipynb`, executes top-to-bottom via `nbclient`, and checks SHA-256 reproducibility
- `scripts/build_package.py` — Validates a submission file and packs `submission.csv`, `notebook.ipynb`, `README.md`, `.env` into `NeuraHack_Preli_Shield.zip` with SHA-256 verification
- `scripts/build_kaggle_phase3_nb.py` — Generates and syntax-checks `kaggle_phase3.ipynb`

### `data/`
- `data/synthetic.csv` — 6,000 rows (Phase 2)
- `data/synthetic_v2.csv` — 15,000 rows (Phase 3A; upload this + `sample.csv` to Kaggle as `shield-synth-v2` dataset)

---

## 3. Progress & Verified Benchmark History

### Phase 1: Baseline (`src/baseline.py` $\rightarrow$ `submission.csv`)
- **86-row `sample.csv` 5-fold Stratified CV** *(optimistic due to near-duplicate templates)*:
  - Task A Macro F1: `0.5828` | Task B Macro F1: `0.4463` | Task C Macro F1: `0.4940`
- **Verified invariants in `sample.csv`:**
  - `Label 0 (n=32)`: 100% `techniques="NONE"`, 100% `action=0`
  - `Label 5 (n=12)`: 100% `techniques="NONE"`, 100% `action=0`
  - Note: 6 rows in `sample.csv` have unsorted `techniques` strings (e.g. `T4|T3`), while our post-processing outputs sorted `|`-joined codes.

### Phase 2 & 2.5: First Synthetic Data + Leakage Diagnostic
- `data/synthetic.csv` (6,000 rows, 3–7 scenarios/class) initially scored `0.9474` on `sample.csv` (`Synth-only -> Real`), **but** Phase 2.5 diagnostic revealed severe template leakage:
  - `63.95%` (55/86) of `sample.csv` rows had a synthetic neighbor with cosine similarity $\ge 0.6$ (`45.35%` $\ge 0.8$).
  - Dropping leaked families ($\text{sim} \ge 0.6$) dropped `Synth-only -> Real` Task A Macro F1 to `0.3558`.
  - Scenario-held-out `StratifiedGroupKFold` scored `0.3604` (clean) / `0.3307` (noisy).

### Phase 3A: Expanded Scenario Diversity (`src/synth_v2.py` $\rightarrow$ `data/synthetic_v2.csv`, `submission_phase3a.csv`)
- Expanded to **15–21 distinct scenarios per class**:
  - Class 0: `21` | Class 1: `16` | Class 2: `15` | Class 3: `15` | Class 4: `15` | Class 5: `20`
  - 15,000 rows total: ~25% per language (`english`: 3,883, `banglish`: 3,772, `mixed`: 3,769, `bangla`: 3,576), `31.48%` multi-turn, `30.49%` noisy, plus `860` `real_paraphrase` rows (`10` per `sample.csv` row).
- **Scenario-Held-Out `StratifiedGroupKFold` (`scenario_id`):**
  - **All v2 (`n=15,000`):** CLEAN `A=0.8780, B=0.7700, C=0.8722` | NOISY `A=0.8624, B=0.7383, C=0.8535`
  - **Synthetic scenarios only (`n=14,140`):** CLEAN `A=0.8735, B=0.7671, C=0.8676` | NOISY `A=0.8571, B=0.7353, C=0.8490`
- **Honest Real-Row Check (`src/phase3a_check.py`):**
  - **(1A) Train on `source == "synthetic_v2"` only (`n=13,168`) $\rightarrow$ 86 real rows:**
    - CLEAN: `A=0.8370, B=0.7085, C=0.8145`
    - NOISY: `A=0.8034, B=0.6958, C=0.8396`
  - **(1B) 5-fold over 86 real rows (`synthetic_v2` + train-fold `real_paraphrase` only):**
    - CLEAN: `A=0.8771, B=0.7144, C=0.8428`
    - NOISY: `A=0.7771, B=0.6838, C=0.8106`
  - *Main error pattern:* Classes 1–5 have high recall, while remaining errors are mostly hard legitimate Class 0 rows (e.g., OTP warnings, personal transfers) false-positiving as Class 1/4/5.
- **Cross-Scenario Cosine Leakage (`n=14,140`):** Mean `0.3470`, Median `0.3411`, $\ge 0.8$: `0.15%` (`21/14,140`).
- **Test Label Distribution (`submission_phase3a.csv`, 500,000 rows, `PASS`):**
  - `0`: 156,366 (31.27%) | `1`: 92,314 (18.46%) | `2`: 51,580 (10.32%) | `3`: 48,105 (9.62%) | `4`: 65,301 (13.06%) | `5`: 86,334 (17.27%)

---

## 4. Exact Next Steps (Where We Left Off)
1. **Run Phase 3B on Kaggle GPU (`kaggle_phase3.ipynb`):**
   - Upload `data/synthetic_v2.csv` + `sample.csv` to Kaggle as private dataset `shield-synth-v2`.
   - Upload `kaggle_phase3.ipynb` to Kaggle with **GPU (T4/P100)** and **Internet ON**, attach `nsucec-neurahack-2026-preliminary` + `shield-synth-v2`, and run all cells.
   - Review the final summary table (Splits 1, 2, 3 across `TF-IDF+LR`, `Frozen E5-base`, `Frozen MiniLM`, and `Fine-tuned xlm-roberta-base`), the 15 real-row errors, and the 10k-row throughput extrapolation.
   - Save `/kaggle/working/*.npz` held-out probabilities and model weights.
2. **Phase 4 — Ensembling & Confidence Calibration (Task D):**
   - Blend held-out probabilities from the top complementary models (e.g. `xlm-roberta-base` + `E5-base` + `TF-IDF+LR`).
   - Tune Task B (`T1–T6`) per-label thresholds on OOF/held-out probabilities.
   - Fit **out-of-fold confidence calibration** for `P(Task A correct)` (logistic/isotonic/temperature scaling fit on OOF predictions, no hardcoded fixed `T`) to minimize Brier Score.
   - Address Class 0 false positives (hard legitimate examples) if still present in the ensemble's top 15 errors.
3. **Phase 5 — Full 500k Test Prediction & Final Packaging:**
   - Run the final calibrated ensemble on all 500,000 rows of `test.csv`, validate with `src/validate_submission.py`, and submit to Kaggle before **27 Nov 6:00 PM BST** (select 2 final submissions!).
   - Sync `notebook.ipynb` and `README.md` with the final pipeline + pretrained model declarations, ensure `submission.csv` matches the selected Kaggle submission bit-for-bit, and run `python scripts/build_package.py` to produce `NeuraHack_Preli_Shield.zip` before **27 Nov 10:00 PM BST**.
