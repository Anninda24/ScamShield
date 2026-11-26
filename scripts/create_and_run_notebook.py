import hashlib
import os
import nbformat as nbf
from nbclient import NotebookClient


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_src(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. Title & Declaration Markdown Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# ScamShield BD — Phase 2 Solution (Synthetic Data, Noise Augmentation & Honest Validation)\n"
        "**Team Name:** Shield\n\n"
        "## Declaration\n"
        "- **Required Libraries & Versions:** `python==3.14.1`, `numpy==2.5.1`, `pandas==3.1.1`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `python-dotenv==1.2.4`\n"
        "- **External Datasets:** None. Training uses only `sample.csv` plus programmatically generated synthetic/noisy data.\n"
        "- **Pretrained Models:** None.\n"
        "- **APIs Used:** None (no API calls; `.env` is empty).\n"
        "- **Data Generation & Augmentation:** Compositional slot-filling generator (`6,000` rows across English/Bangla/Banglish/Mixed, ~29% multi-turn, hard-negative pairs, real + invented brands) + adversarial noise augmenter (`~30%` of synthetic rows + noisy copies of the `86` real rows) using `SEED = 42`. No rows, labels, or predictions from `test.csv` are ever used in training or augmentation.\n"
        "- **Methodology & Validation Strategy:**\n"
        "  - *Preprocessing:* Unicode NFKC normalization; removal of `\\u200b`, `\\u200c`, `\\ufeff` (preserving `\\u200d` ZWJ for Bangla); Cyrillic/Greek-to-Latin homoglyph mapping; Latin-only lowercasing; defanged URL replacement (`[URL]`); digit-run replacement (`[NUM]`); `|||` separator preservation.\n"
        "  - *Features:* `TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True)` + `TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True)` combined via `scipy.sparse.hstack`.\n"
        "  - *Models:* Task A (`LogisticRegression(class_weight='balanced')`), Task B (`OneVsRestClassifier(LogisticRegression(class_weight='balanced'))`, threshold `0.5`), Task C (`LogisticRegression(class_weight='balanced')`) trained on real (`5x` weight) + noisy real (`3x` weight) + synthetic (`1x` weight).\n"
        "  - *Post-Processing:* `label == 0 -> techniques = 'NONE', action = 0`; non-zero labels with empty technique sets fall back to the single highest-probability technique; sorted `|`-joined codes; confidence = max predicted Task A probability clipped to `[0.05, 0.98]` and rounded to 4 decimals.\n"
        "  - *Honest Validation:* Evaluated on CLEAN and NOISY sets across (a) real-only 5-fold CV, (b) synthetic-only -> real 86 rows, (c) 5-fold CV on real with synthetic in train folds, and (d) `StratifiedGroupKFold` on synthetic grouped by `family_id`.\n"
        "- **Random Seeds:** `SEED = 42` for `random`, `numpy`, and `sklearn`.\n"
        "- **How to Reproduce:** Place `sample.csv`, `test.csv`, and `sample_submission.csv` in the working directory (or `./data/`) and run all cells top to bottom.\n"
    ))

    # 2. First Code Cell: Imports & Seed 42
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import re\n"
        "import time\n"
        "import random\n"
        "import unicodedata\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy.sparse import hstack\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.multiclass import OneVsRestClassifier\n"
        "from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold\n"
        "from sklearn.metrics import f1_score, confusion_matrix\n"
        "from dotenv import load_dotenv\n"
        "\n"
        "load_dotenv()\n"
        "\n"
        "SEED = 42\n"
        "random.seed(SEED)\n"
        "np.random.seed(SEED)\n"
        "\n"
        "DATA_DIR = 'data' if os.path.isfile(os.path.join('data', 'sample.csv')) else '.'\n"
        "TECH_CODES = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']\n"
        "print('Environment ready. SEED =', SEED, '| DATA_DIR =', DATA_DIR)\n"
    ))

    # 3. Inline src/preprocess.py
    cells.append(nbf.v4.new_markdown_cell("## 1. Preprocessing"))
    prep_code = read_src("src/preprocess.py").split("if __name__")[0].strip()
    cells.append(nbf.v4.new_code_cell(prep_code + "\n\nrun_unit_tests()\n"))

    # 4. Inline src/noise.py
    cells.append(nbf.v4.new_markdown_cell("## 2. Noise Augmenter"))
    noise_code = read_src("src/noise.py").split("if __name__")[0].strip()
    cells.append(nbf.v4.new_code_cell(noise_code))

    # 5. Inline src/synth.py
    cells.append(nbf.v4.new_markdown_cell("## 3. Compositional Synthetic Data Generator"))
    synth_code = read_src("src/synth.py").split("if __name__")[0].strip()
    cells.append(nbf.v4.new_code_cell(synth_code))

    # 6. Inline Phase 2 Honest Validation, Training & Prediction
    cells.append(nbf.v4.new_markdown_cell("## 4. Honest Validation, Phase 2 Training, Prediction & Validation"))
    pipe_code = read_src("src/phase2_pipeline.py")
    # Remove local module imports since everything is inlined above
    lines = []
    for line in pipe_code.splitlines():
        if line.startswith("from preprocess import") or line.startswith("from synth import") or line.startswith("from noise import") or line.startswith("from validate_submission import"):
            continue
        lines.append(line)
    # Inline validate_submission helper
    val_code = read_src("src/validate_submission.py").split("if __name__")[0].strip()
    inline_val = val_code.replace("data_dir = get_data_dir()", "data_dir = DATA_DIR")

    cells.append(nbf.v4.new_code_cell(inline_val + "\n\n" + "\n".join(lines)))

    nb["cells"] = cells
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created Phase 2 notebook.ipynb. Executing top to bottom...")

    pre_hash = sha256_file("submission_phase2.csv") if os.path.isfile("submission_phase2.csv") else None
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    post_hash = sha256_file("submission_phase2.csv")
    print("Notebook executed cleanly! submission_phase2.csv SHA-256:", post_hash)
    if pre_hash:
        print("Reproducibility check (matches src/phase2_pipeline.py submission_phase2.csv):", pre_hash == post_hash)


if __name__ == "__main__":
    build_notebook()
