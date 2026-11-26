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


def strip_local_imports(code: str) -> str:
    prefixes = (
        "from preprocess import",
        "from synth import",
        "from synth_v2 import",
        "from noise import",
        "from validate_submission import",
        "from phase2_pipeline import",
    )
    lines = []
    skip_block = False
    for line in code.splitlines():
        if any(line.startswith(p) for p in prefixes):
            if "(" in line and ")" not in line:
                skip_block = True
            continue
        if skip_block:
            if ")" in line:
                skip_block = False
            continue
        lines.append(line)
    return "\n".join(lines)


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. Title & Declaration Markdown Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# ScamShield BD — Phase 3A Solution (Expanded Scenarios, Paraphrases & Cross-Scenario Check)\n"
        "**Team Name:** Shield\n\n"
        "## Declaration\n"
        "- **Required Libraries & Versions:** `python==3.14.1`, `numpy==2.5.1`, `pandas==3.1.1`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `python-dotenv==1.2.4`\n"
        "- **External Datasets:** None. Training uses only `sample.csv` plus programmatically generated synthetic/noisy/paraphrased data.\n"
        "- **Pretrained Models:** None.\n"
        "- **APIs Used:** None (no API calls; `.env` is empty).\n"
        "- **Data Generation & Augmentation:**\n"
        "  - Expanded compositional generator (`data/synthetic_v2.csv`, `15,000` rows): 21 scenarios in Class 0, 16 in Class 1, 15 in Class 2, 15 in Class 3, 15 in Class 4, 20 in Class 5 across English/Bangla/Banglish/Mixed (~25% each), ~31.5% multi-turn, ~30.5% noisy (`apply_noise`), hard-negative contrast twins, plus `860` paraphrase variants (`10` per `sample.csv` row, `source='real_paraphrase'`).\n"
        "  - No rows, labels, or predictions from `test.csv` are ever used in training or augmentation.\n"
        "- **Methodology & Validation Strategy:**\n"
        "  - *Preprocessing:* Unicode NFKC normalization; removal of `\\u200b`, `\\u200c`, `\\ufeff` (preserving `\\u200d` ZWJ for Bangla); Cyrillic/Greek-to-Latin homoglyph mapping; Latin-only lowercasing; `[URL]` and `[NUM]` tokens; `|||` preservation.\n"
        "  - *Features & Models:* `char_wb` `(2, 5)` + `word` `(1, 2)` TF-IDF (`sublinear_tf=True`) + `LogisticRegression(class_weight='balanced')` for Task A & C, `OneVsRestClassifier(LogisticRegression(class_weight='balanced'))` for Task B.\n"
        "  - *Validation:* Scenario-held-out `StratifiedGroupKFold`, real-row check (`synthetic_v2` only & 5-fold with train-fold paraphrases only), and cross-scenario cosine leakage check.\n"
        "  - *Final Model:* Trained on `sample.csv` (`5x` weight) + `data/synthetic_v2.csv` (`1x` weight), outputting `submission_phase3a.csv`.\n"
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
        "from sklearn.preprocessing import normalize\n"
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

    # 3. Preprocessing
    cells.append(nbf.v4.new_markdown_cell("## 1. Preprocessing"))
    prep_code = read_src("src/preprocess.py").split("if __name__")[0].strip()
    cells.append(nbf.v4.new_code_cell(prep_code + "\n\nrun_unit_tests()\n"))

    # 4. Noise Augmenter
    cells.append(nbf.v4.new_markdown_cell("## 2. Noise Augmenter"))
    noise_code = read_src("src/noise.py").split("if __name__")[0].strip()
    cells.append(nbf.v4.new_code_cell(noise_code))

    # 5. V1 + V2 Synthetic Generators
    cells.append(nbf.v4.new_markdown_cell("## 3. Expanded Compositional Synthetic Generator (`synth_v2`)"))
    synth_v1_code = read_src("src/synth.py").split("if __name__")[0].strip()
    synth_v2_code = strip_local_imports(read_src("src/synth_v2.py").split("if __name__")[0].strip())
    cells.append(nbf.v4.new_code_cell(synth_v1_code + "\n\n" + synth_v2_code))

    cells.append(nbf.v4.new_code_cell(
        "df_v2 = generate_synthetic_v2(target_rows=15000, seed=SEED)\n"
        "os.makedirs('data', exist_ok=True)\n"
        "df_v2.to_csv('data/synthetic_v2.csv', index=False)\n"
        "print('Generated data/synthetic_v2.csv:', df_v2.shape)\n"
    ))

    # 6. Helpers + Validator + Phase 3A Check & Prediction
    cells.append(nbf.v4.new_markdown_cell("## 4. Validation, Phase 3A Checks, & `submission_phase3a.csv`"))
    val_code = read_src("src/validate_submission.py").split("if __name__")[0].strip().replace("data_dir = get_data_dir()", "data_dir = DATA_DIR")
    p2_helpers = strip_local_imports(read_src("src/phase2_pipeline.py").split("def run_phase2()")[0].strip())
    p3a_code = strip_local_imports(read_src("src/phase3a_check.py"))
    cells.append(nbf.v4.new_code_cell(val_code + "\n\n" + p2_helpers + "\n\n" + p3a_code))

    nb["cells"] = cells
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created Phase 3A notebook.ipynb. Executing top to bottom...")

    pre_hash = sha256_file("submission_phase3a.csv") if os.path.isfile("submission_phase3a.csv") else None
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.execute()
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    post_hash = sha256_file("submission_phase3a.csv")
    print("Notebook executed cleanly! submission_phase3a.csv SHA-256:", post_hash)
    if pre_hash:
        print("Reproducibility check (matches src/phase3a_check.py submission_phase3a.csv):", pre_hash == post_hash)


if __name__ == "__main__":
    build_notebook()
