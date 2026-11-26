# ScamShield BD — NSUCEC NeuraHack 2026 Preliminary Round

## 1. Team Information
- **Team Name:** Shield
- **Members:** Asraful Islam Afridi, Linkon Biswas, Anninda Raj Das
- **University:** Gopalganj Science and Technology University

## 2. Approach Summary (Phase 1 Baseline)
- **Preprocessing:** Unicode NFKC normalization; removal of zero-width characters `\u200b`, `\u200c`, and BOM `\ufeff` while preserving `\u200d` (ZWJ) for Bangla; Cyrillic/Greek-to-Latin homoglyph mapping; Latin-only lowercasing; replacement of defanged URLs with `[URL]` and digit runs with `[NUM]`; preservation of the `|||` conversation turn separator; whitespace collapsing.
- **Features:** Concatenation (`scipy.sparse.hstack`) of character n-gram TF-IDF (`char_wb`, n-grams 2–5, `sublinear_tf=True`) and word n-gram TF-IDF (n-grams 1–2, `sublinear_tf=True`).
- **Models:**
  - **Task A (6-class fraud classification):** `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`
  - **Task B (multi-label T1–T6):** `OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))` with threshold `0.5`
  - **Task C (intended victim action 0–5):** `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`
- **Post-Processing:** If `label == 0`, set `techniques = "NONE"` and `action = 0`. For any non-zero label with an empty predicted technique set, assign the single highest-probability technique. Sort technique codes and join with `"|"`.
- **Confidence (Task D):** Max predicted probability from Task A, clipped to `[0.05, 0.98]` and rounded to 4 decimal places.

## 3. External Models, Datasets, and Resources Used
- **External Datasets:** None (trained strictly on `sample.csv`).
- **Pretrained Models:** None.
- **APIs / Other Resources:** None.

## 4. Random Seeds
- `SEED = 42` fixed across `random`, `numpy`, and `scikit-learn` (`random_state=42`).

## 5. Environment
- **Python:** `3.14.1`
- **Libraries:**
  - `numpy==2.5.1`
  - `pandas==3.1.1`
  - `scipy==1.18.1`
  - `scikit-learn==1.9.1`
  - `python-dotenv==1.2.4`
- **Hardware:** Local Windows CPU (baseline runs on CPU; no GPU required for TF-IDF + LogisticRegression)
- **Expected Runtime:** ~90 seconds end-to-end for training on `sample.csv` and predicting all 500,000 rows of `test.csv`.

## 6. Instructions for Running the Notebook
1. Ensure `sample.csv`, `test.csv`, and `sample_submission.csv` are in the same directory as `notebook.ipynb` (or inside a `./data/` subfolder).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `.env` is present in the same directory (an empty `.env` file is included as no API keys are used).
4. Open `notebook.ipynb` in Jupyter/VS Code/Antigravity and select **Restart & Run All**, or execute from CLI:
   ```bash
   python scripts/create_and_run_notebook.py
   ```
5. The notebook will regenerate and validate `submission.csv` (500,000 rows).
