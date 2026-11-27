import ast
import os
import nbformat as nbf


def read_src(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def strip_local_imports(code: str) -> str:
    prefixes = (
        "from preprocess import",
        "from synth import",
        "from synth_v2 import",
        "from synth_v3 import",
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


def build_final_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. Declaration Markdown Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# ScamShield BD — Final Reproducible Solution (`submission_v3.csv`)\n"
        "**Team Name:** Shield\n\n"
        "## Declaration\n"
        "- **Required Libraries & Versions:** `python>=3.10`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch` (CUDA), `transformers`, `python-dotenv` (run on Kaggle GPU T4/P100 environment; local dev used `python==3.14.1`, `numpy==2.5.1`, `pandas==3.1.1`, `scipy==1.18.1`, `scikit-learn==1.9.1`).\n"
        "- **External Pretrained Model:** `xlm-roberta-base` from HuggingFace Hub (MIT License).\n"
        "- **External Datasets / APIs / Pseudo-labeling:** None. No external datasets, no hosted LLM APIs, no pseudo-labeling, and no `test.csv` labels or texts are used in training or data generation.\n"
        "- **Random Seeds:** `42` (primary seed for `random`, `numpy`, `sklearn`, `torch`, `transformers`, synthetic generation, TF-IDF, and XLM-R seed 1) and `7` (XLM-R seed 2).\n"
        "- **Methodology:**\n"
        "  - *Preprocessing:* Unicode NFKC normalization; removal of `\\u200b`, `\\u200c`, `\\ufeff` (preserving `\\u200d` ZWJ for Bangla); Cyrillic/Greek homoglyph mapping; Latin lowercasing; `[URL]` & `[NUM]` tokens; `|||` preservation.\n"
        "  - *Data Generation & Augmentation:* Programmatically regenerates `synthetic_v3.csv` (`16,107` rows) from scratch using `src/synth.py`, `src/synth_v2.py`, `src/synth_v3.py`, and `src/noise.py` (`~30%` noisy), plus `300` programmatic noise-only rows (`label=0, NONE, 0`), plus `86` real rows from `sample.csv` (clean & noisy copies at `5x` weight).\n"
        "  - *Models:* Two 3-head `xlm-roberta-base` models (`seed=42` and `seed=7`; loss `0.5*L_A + 0.3*L_B + 0.2*L_C`, `fp16`, `max_len=256` head+tail truncation, `batch_size=32`, `3 epochs`, `lr=2e-5`, `10%` warmup) + `char_wb` `(2,5)` & `word` `(1,2)` TF-IDF + `LogisticRegression(class_weight='balanced')`.\n"
        "  - *Blend, Thresholds & Calibration:* Held-out Split 2 (86 real rows clean/noisy) + Split 1 Fold 0 determine blend weights (`0.5` preferred unless `> 0.02` gain), Task B thresholds (`0.5` unless `> 0.01` gain on both held-out sets), and lowest-Brier confidence calibrator (raw vs temperature vs isotonic vs Platt) clipped to `[0.05, 0.98]`.\n"
        "  - *Post-Processing:* `label in {0, 5} -> techniques='NONE', action=0`; non-empty technique fallback for `1–4`; OOV noise-only rule (`label=0, NONE, 0, conf=0.5`).\n"
        "- **Validation Strategy & Caveats:** Scenario-grouped 5-fold CV (`StratifiedGroupKFold` on `scenario_id`) and real-row check (training on `synthetic_v2` without paraphrases, evaluating on `86` real rows). Caveat: `86` real rows is a small evaluation set and synthetic data may not cover all unseen test phrasing.\n"
        "- **Runtime & Hardware:** Kaggle GPU (`T4`/`P100`), ~35–55 min total (data gen `<15s`, held-out Split 2 & Split 1 F0 `~10m`, 2-seed final XLM-R training `~15-20m`, 500k FP16 length-sorted inference `~12-18m`).\n"
        "- **Determinism Note:** `fp16` GPU training on CUDA is not bit-for-bit deterministic across hardware/driver runs, so re-runs can differ slightly.\n"
        "- **AI Coding Assistants:** AI coding assistant (Antigravity) was used to assist in writing code.\n"
    ))

    # 2. Imports, Seeds, Recursive Path Detection
    cells.append(nbf.v4.new_code_cell(
        "import gc\n"
        "import glob\n"
        "import os\n"
        "import pickle\n"
        "import re\n"
        "import time\n"
        "import random\n"
        "import unicodedata\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.data import Dataset, DataLoader\n"
        "from scipy.sparse import hstack\n"
        "from scipy.optimize import minimize_scalar\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.multiclass import OneVsRestClassifier\n"
        "from sklearn.model_selection import StratifiedGroupKFold\n"
        "from sklearn.isotonic import IsotonicRegression\n"
        "from sklearn.metrics import f1_score, brier_score_loss\n"
        "from sklearn.metrics.pairwise import cosine_similarity\n"
        "from sklearn.utils.class_weight import compute_class_weight\n"
        "from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed as hf_set_seed\n\n"
        "SEED = 42\n"
        "def seed_everything(seed=42):\n"
        "    random.seed(seed)\n"
        "    np.random.seed(seed)\n"
        "    torch.manual_seed(seed)\n"
        "    torch.cuda.manual_seed_all(seed)\n"
        "    hf_set_seed(seed)\n\n"
        "seed_everything(SEED)\n"
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "WORK_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'\n"
        "os.makedirs(WORK_DIR, exist_ok=True)\n\n"
        "# Recursive competition folder detection (must contain both sample.csv and test.csv)\n"
        "SEARCH_ROOT = '/kaggle/input' if os.path.isdir('/kaggle/input') else '.'\n"
        "COMP_DIR = None\n"
        "for root, _, files in os.walk(SEARCH_ROOT):\n"
        "    if 'sample.csv' in files and 'test.csv' in files:\n"
        "        COMP_DIR = root\n"
        "        break\n"
        "if COMP_DIR is None:\n"
        "    raise FileNotFoundError(f'Could not find folder containing sample.csv and test.csv under {SEARCH_ROOT}')\n\n"
        "print(f'DEVICE={DEVICE} | COMP_DIR={COMP_DIR} | WORK_DIR={WORK_DIR}')\n"
    ))

    # 3. Preprocessing & Noise Augmenter
    cells.append(nbf.v4.new_markdown_cell("## 1. Preprocessing & Noise Augmentation (`preprocess.py` & `noise.py`)"))
    prep_code = read_src("src/preprocess.py").split("if __name__")[0].strip()
    noise_code = read_src("src/noise.py").split("if __name__")[0].strip()
    cells.append(nbf.v4.new_code_cell(prep_code + "\n\nrun_unit_tests()\n\n" + noise_code))

    # 4. Data Generation from scratch (synth.py + synth_v2.py + synth_v3.py)
    cells.append(nbf.v4.new_markdown_cell("## 2. Programmatic Synthetic Data Generation from Scratch (`synth.py`, `synth_v2.py`, `synth_v3.py`)"))
    synth1 = read_src("src/synth.py").split("if __name__")[0].strip()
    synth2 = strip_local_imports(read_src("src/synth_v2.py").split("if __name__")[0].strip())
    synth2 = synth2.replace('pd.read_csv("sample.csv")', "pd.read_csv(os.path.join(COMP_DIR, 'sample.csv'))")
    synth3 = strip_local_imports(read_src("src/synth_v3.py").split("if __name__")[0].strip())
    synth3 = synth3.replace('pd.read_csv("sample.csv")', "pd.read_csv(os.path.join(COMP_DIR, 'sample.csv'))")
    synth3 = synth3.replace('pd.read_csv("data/synthetic_v2.csv", keep_default_na=False)', 'v2_df.copy()')
    synth3 = synth3.replace('os.makedirs("data", exist_ok=True)', "os.makedirs(WORK_DIR, exist_ok=True)")
    synth3 = synth3.replace('v3_df.to_csv("data/synthetic_v3.csv", index=False)', "v3_df.to_csv(os.path.join(WORK_DIR, 'synthetic_v3.csv'), index=False)")
    synth3 = synth3.replace("def generate_v3_patch(seed: int = SEED) -> pd.DataFrame:", "def generate_v3_patch(v2_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:")

    cells.append(nbf.v4.new_code_cell(
        synth1 + "\n\n" + synth2 + "\n\n" + synth3 + "\n\n"
        "df_v2_gen = generate_synthetic_v2(target_rows=15000, seed=SEED)\n"
        "generate_v3_patch(df_v2_gen, seed=SEED)\n"
        "synth_df = df_v2_gen.copy()\n"
        "v3_df = pd.read_csv(os.path.join(WORK_DIR, 'synthetic_v3.csv'), keep_default_na=False)\n"
        "print('Regenerated synthetic_v3.csv from scratch:', v3_df.shape)\n"
    ))

    # 5. Noise-only generator, splits, ThreeHeadEncoder, Held-out Split 2 & Split 1 F0 for blend/thresholds/calibration
    cells.append(nbf.v4.new_markdown_cell("## 3. Held-Out Split 2 & Split 1 F0 (Blend Weights, Task B Thresholds & Calibration)"))
    cells.append(nbf.v4.new_code_cell(
        "TECH_CODES = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']\n"
        "BACKBONE = 'xlm-roberta-base'\n\n"
        "def encode_techniques(series):\n"
        "    y = np.zeros((len(series), len(TECH_CODES)), dtype=int)\n"
        "    for i, val in enumerate(series.astype(str)):\n"
        "        parts = val.split('|')\n"
        "        for j, code in enumerate(TECH_CODES):\n"
        "            if code in parts:\n"
        "                y[i, j] = 1\n"
        "    return y\n\n"
        "real_df = pd.read_csv(os.path.join(COMP_DIR, 'sample.csv'))\n"
        "rng = random.Random(SEED)\n"
        "real_df['clean_text'] = real_df['text'].fillna('').astype(str).apply(preprocess_text)\n"
        "real_df['noisy_text'] = [preprocess_text(apply_noise(t, rng, level=0.35)) for t in real_df['text'].fillna('').astype(str)]\n"
        "synth_df['clean_text'] = synth_df['text'].fillna('').astype(str).apply(preprocess_text)\n"
        "synth_df['noisy_text'] = [preprocess_text(apply_noise(t, rng, level=0.35)) for t in synth_df['text'].fillna('').astype(str)]\n"
        "v3_df['clean_text'] = v3_df['text'].fillna('').astype(str).apply(preprocess_text)\n\n"
        "real_lbl = real_df['label'].astype(int).values\n"
        "real_tech = encode_techniques(real_df['techniques'])\n"
        "real_act = real_df['action'].astype(int).values\n"
        "synth_lbl = synth_df['label'].astype(int).values\n"
        "synth_tech = encode_techniques(synth_df['techniques'])\n"
        "synth_act = synth_df['action'].astype(int).values\n\n"
        "s1_mask = (synth_df['source'] != 'real_paraphrase').values\n"
        "s1_indices = np.where(s1_mask)[0]\n"
        "sgkf = StratifiedGroupKFold(n_splits=5)\n"
        "split1_folds = [(s1_indices[tr], s1_indices[va]) for tr, va in sgkf.split(s1_indices, synth_lbl[s1_mask], synth_df.loc[s1_mask, 'scenario_id'].values)]\n"
        "split2_train_idx = np.where(synth_df['source'] == 'synthetic_v2')[0]\n"
        "tr_f0, va_f0 = split1_folds[0]\n\n"
        "class MultiTaskDataset(Dataset):\n"
        "    def __init__(self, texts, lbls, techs, acts, weights, tokenizer, max_len=256):\n"
        "        self.texts, self.lbls, self.techs, self.acts, self.weights, self.tok, self.max_len = texts, lbls, techs, acts, weights, tokenizer, max_len\n"
        "    def __len__(self):\n"
        "        return len(self.texts)\n"
        "    def __getitem__(self, idx):\n"
        "        ids = self.tok.encode(self.texts[idx], add_special_tokens=False)\n"
        "        max_body = self.max_len - 2\n"
        "        if len(ids) > max_body:\n"
        "            half = max_body // 2\n"
        "            ids = ids[:half] + ids[-(max_body - half):]\n"
        "        return {'input_ids': [self.tok.cls_token_id] + ids + [self.tok.sep_token_id],\n"
        "                'lbl': int(self.lbls[idx]), 'tech': self.techs[idx].astype(np.float32),\n"
        "                'act': int(self.acts[idx]), 'w': float(self.weights[idx])}\n\n"
        "def make_collate_fn(pad_id):\n"
        "    def collate(batch):\n"
        "        max_l = max(len(b['input_ids']) for b in batch)\n"
        "        inp, mask = [], []\n"
        "        for b in batch:\n"
        "            l = len(b['input_ids'])\n"
        "            inp.append(b['input_ids'] + [pad_id] * (max_l - l))\n"
        "            mask.append([1] * l + [0] * (max_l - l))\n"
        "        return {'input_ids': torch.tensor(inp, dtype=torch.long), 'attention_mask': torch.tensor(mask, dtype=torch.long),\n"
        "                'lbl': torch.tensor([b['lbl'] for b in batch], dtype=torch.long),\n"
        "                'tech': torch.tensor(np.array([b['tech'] for b in batch]), dtype=torch.float32),\n"
        "                'act': torch.tensor([b['act'] for b in batch], dtype=torch.long),\n"
        "                'w': torch.tensor([b['w'] for b in batch], dtype=torch.float32)}\n"
        "    return collate\n\n"
        "class ThreeHeadEncoder(nn.Module):\n"
        "    def __init__(self, model_name='xlm-roberta-base'):\n"
        "        super().__init__()\n"
        "        self.backbone = AutoModel.from_pretrained(model_name)\n"
        "        h = self.backbone.config.hidden_size\n"
        "        self.drop = nn.Dropout(0.15)\n"
        "        self.head_a = nn.Linear(h, 6)\n"
        "        self.head_b = nn.Linear(h, 6)\n"
        "        self.head_c = nn.Linear(h, 6)\n"
        "    def forward(self, input_ids, attention_mask):\n"
        "        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state\n"
        "        mask = attention_mask.unsqueeze(-1).expand(out.size()).float()\n"
        "        pooled = torch.sum(out * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)\n"
        "        x = self.drop(pooled)\n"
        "        return self.head_a(x), self.head_b(x), self.head_c(x)\n\n"
        "@torch.no_grad()\n"
        "def predict_three_head(model, texts, tokenizer, batch_size=64, max_len=256):\n"
        "    model.eval()\n"
        "    ds = MultiTaskDataset(texts, np.zeros(len(texts)), np.zeros((len(texts), 6)), np.zeros(len(texts)), np.ones(len(texts)), tokenizer, max_len)\n"
        "    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer.pad_token_id))\n"
        "    pa_list, pb_list, pc_list = [], [], []\n"
        "    for batch in dl:\n"
        "        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda'), dtype=torch.float16):\n"
        "            la, lb, lc = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))\n"
        "        pa_list.append(torch.softmax(la.float(), dim=-1).cpu().numpy())\n"
        "        pb_list.append(torch.sigmoid(lb.float()).cpu().numpy())\n"
        "        pc_list.append(torch.softmax(lc.float(), dim=-1).cpu().numpy())\n"
        "    return np.vstack(pa_list), np.vstack(pb_list), np.vstack(pc_list)\n\n"
        "def train_three_head(run_name, tr_texts, tr_lbl, tr_tech, tr_act, tr_weights, ckpt_path, model_name='xlm-roberta-base', epochs=3, batch_size=32, lr=2e-5, max_len=256):\n"
        "    tok = AutoTokenizer.from_pretrained(model_name)\n"
        "    model = ThreeHeadEncoder(model_name).to(DEVICE)\n"
        "    if os.path.isfile(ckpt_path):\n"
        "        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))\n"
        "        return model, tok\n"
        "    ds = MultiTaskDataset(tr_texts, tr_lbl, tr_tech, tr_act, tr_weights, tok, max_len)\n"
        "    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(tok.pad_token_id))\n"
        "    cw = compute_class_weight('balanced', classes=np.arange(6), y=tr_lbl)\n"
        "    ce_a = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE), reduction='none')\n"
        "    bce_b = nn.BCEWithLogitsLoss(reduction='none')\n"
        "    ce_c = nn.CrossEntropyLoss(reduction='none')\n"
        "    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)\n"
        "    total_steps = len(dl) * epochs\n"
        "    sched = get_linear_schedule_with_warmup(opt, int(0.10 * total_steps), total_steps)\n"
        "    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))\n"
        "    for ep in range(epochs):\n"
        "        t_ep = time.time()\n"
        "        model.train()\n"
        "        tot_loss = 0.0\n"
        "        for batch in dl:\n"
        "            opt.zero_grad()\n"
        "            inp, msk = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)\n"
        "            la_y, lb_y, lc_y, sw = batch['lbl'].to(DEVICE), batch['tech'].to(DEVICE), batch['act'].to(DEVICE), batch['w'].to(DEVICE)\n"
        "            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda'), dtype=torch.float16):\n"
        "                la, lb, lc = model(inp, msk)\n"
        "                loss = 0.5 * (ce_a(la, la_y) * sw).mean() + 0.3 * (bce_b(lb, lb_y).mean(dim=1) * sw).mean() + 0.2 * (ce_c(lc, lc_y) * sw).mean()\n"
        "            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()\n"
        "            tot_loss += loss.item()\n"
        "        print(f'[{run_name}] Epoch {ep+1}/{epochs} | loss={tot_loss/len(dl):.4f} | time={time.time()-t_ep:.1f}s')\n"
        "    torch.save(model.state_dict(), ckpt_path)\n"
        "    return model, tok\n\n"
        "def free_gpu(*var_names):\n"
        "    for vn in var_names:\n"
        "        if vn in globals():\n"
        "            del globals()[vn]\n"
        "    gc.collect()\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.empty_cache()\n"
    ))

    # 6. Held-out blend selection, Task B thresholds, calibration (fit on v2 Split 2 + Split 1 F0)
    cells.append(nbf.v4.new_code_cell(
        "def fit_pred_tfidf_ho(tr_texts, tr_l, tr_t, tr_a, vc, vn):\n"
        "    cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, dtype=np.float32)\n"
        "    wv = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)\n"
        "    X_tr = hstack([cv.fit_transform(tr_texts), wv.fit_transform(tr_texts)]).tocsr()\n"
        "    X_vc = hstack([cv.transform(vc), wv.transform(vc)]).tocsr()\n"
        "    X_vn = hstack([cv.transform(vn), wv.transform(vn)]).tocsr()\n"
        "    ca = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED).fit(X_tr, tr_l)\n"
        "    cb = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)).fit(X_tr, tr_t)\n"
        "    cc = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED).fit(X_tr, tr_a)\n"
        "    return ca.predict_proba(X_vc), cb.predict_proba(X_vc), cc.predict_proba(X_vc), ca.predict_proba(X_vn), cb.predict_proba(X_vn), cc.predict_proba(X_vn)\n\n"
        "tf_s2_pac, tf_s2_pbc, tf_s2_pcc, tf_s2_pan, tf_s2_pbn, tf_s2_pcn = fit_pred_tfidf_ho(\n"
        "    synth_df.loc[split2_train_idx, 'clean_text'].tolist(), synth_lbl[split2_train_idx], synth_tech[split2_train_idx], synth_act[split2_train_idx],\n"
        "    real_df['clean_text'].tolist(), real_df['noisy_text'].tolist()\n"
        ")\n"
        "tf_s1_pac, tf_s1_pbc, tf_s1_pcc, tf_s1_pan, tf_s1_pbn, tf_s1_pcn = fit_pred_tfidf_ho(\n"
        "    synth_df.loc[tr_f0, 'clean_text'].tolist(), synth_lbl[tr_f0], synth_tech[tr_f0], synth_act[tr_f0],\n"
        "    synth_df.loc[va_f0, 'clean_text'].tolist(), synth_df.loc[va_f0, 'noisy_text'].tolist()\n"
        ")\n\n"
        "seed_everything(42)\n"
        "m_s2_42, tok_ho = train_three_head('XLM-R S2 s42', synth_df.loc[split2_train_idx, 'clean_text'].tolist(), synth_lbl[split2_train_idx], synth_tech[split2_train_idx], synth_act[split2_train_idx], np.ones(len(split2_train_idx)), os.path.join(WORK_DIR, 'xlmr_split2.pt'), BACKBONE)\n"
        "x2_42_c = predict_three_head(m_s2_42, real_df['clean_text'].tolist(), tok_ho)\n"
        "x2_42_n = predict_three_head(m_s2_42, real_df['noisy_text'].tolist(), tok_ho)\n"
        "free_gpu('m_s2_42')\n\n"
        "seed_everything(7)\n"
        "m_s2_7, _ = train_three_head('XLM-R S2 s7', synth_df.loc[split2_train_idx, 'clean_text'].tolist(), synth_lbl[split2_train_idx], synth_tech[split2_train_idx], synth_act[split2_train_idx], np.ones(len(split2_train_idx)), os.path.join(WORK_DIR, 'xlmr_split2_seed7.pt'), BACKBONE)\n"
        "x2_7_c = predict_three_head(m_s2_7, real_df['clean_text'].tolist(), tok_ho)\n"
        "x2_7_n = predict_three_head(m_s2_7, real_df['noisy_text'].tolist(), tok_ho)\n"
        "free_gpu('m_s2_7')\n\n"
        "seed_everything(42)\n"
        "m_s1_42, _ = train_three_head('XLM-R S1-F0 s42', synth_df.loc[tr_f0, 'clean_text'].tolist(), synth_lbl[tr_f0], synth_tech[tr_f0], synth_act[tr_f0], np.ones(len(tr_f0)), os.path.join(WORK_DIR, 'xlmr_split1_f0.pt'), BACKBONE)\n"
        "x1_c = predict_three_head(m_s1_42, synth_df.loc[va_f0, 'clean_text'].tolist(), tok_ho)\n"
        "x1_n = predict_three_head(m_s1_42, synth_df.loc[va_f0, 'noisy_text'].tolist(), tok_ho)\n"
        "free_gpu('m_s1_42')\n\n"
        "s2_mean_c = tuple(0.5 * (x2_42_c[i] + x2_7_c[i]) for i in range(3))\n"
        "s2_mean_n = tuple(0.5 * (x2_42_n[i] + x2_7_n[i]) for i in range(3))\n\n"
        "# Blend weights\n"
        "best_w = {}\n"
        "for t_idx, (t_name, tf_s2c, tf_s2n, tf_s1c, tf_s1n) in enumerate([\n"
        "    ('A', tf_s2_pac, tf_s2_pan, tf_s1_pac, tf_s1_pan),\n"
        "    ('B', tf_s2_pbc, tf_s2_pbn, tf_s1_pbc, tf_s1_pbn),\n"
        "    ('C', tf_s2_pcc, tf_s2_pcn, tf_s1_pcc, tf_s1_pcn),\n"
        "]):\n"
        "    scores_w = {}\n"
        "    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:\n"
        "        ps2c = w * x2_42_c[t_idx] + (1-w) * tf_s2c\n"
        "        ps2n = w * x2_42_n[t_idx] + (1-w) * tf_s2n\n"
        "        ps1c = w * x1_c[t_idx] + (1-w) * tf_s1c\n"
        "        ps1n = w * x1_n[t_idx] + (1-w) * tf_s1n\n"
        "        if t_name == 'A':\n"
        "            sc = (f1_score(real_lbl, np.argmax(ps2c,1), average='macro') + f1_score(real_lbl, np.argmax(ps2n,1), average='macro') + f1_score(synth_lbl[va_f0], np.argmax(ps1c,1), average='macro') + f1_score(synth_lbl[va_f0], np.argmax(ps1n,1), average='macro')) / 4.0\n"
        "        elif t_name == 'B':\n"
        "            sc = (np.mean([f1_score(real_tech[:,j], (ps2c[:,j]>0.5).astype(int), zero_division=0) for j in range(6)]) + np.mean([f1_score(real_tech[:,j], (ps2n[:,j]>0.5).astype(int), zero_division=0) for j in range(6)]) + np.mean([f1_score(synth_tech[va_f0,j], (ps1c[:,j]>0.5).astype(int), zero_division=0) for j in range(6)]) + np.mean([f1_score(synth_tech[va_f0,j], (ps1n[:,j]>0.5).astype(int), zero_division=0) for j in range(6)])) / 4.0\n"
        "        else:\n"
        "            sc = (f1_score(real_act, np.argmax(ps2c,1), average='macro') + f1_score(real_act, np.argmax(ps2n,1), average='macro') + f1_score(synth_act[va_f0], np.argmax(ps1c,1), average='macro') + f1_score(synth_act[va_f0], np.argmax(ps1n,1), average='macro')) / 4.0\n"
        "        scores_w[w] = sc\n"
        "    mw = max(scores_w, key=scores_w.get)\n"
        "    best_w[t_name] = mw if (scores_w[mw] - scores_w[0.5] > 0.02) else 0.5\n"
        "wA, wB, wC = best_w['A'], best_w['B'], best_w['C']\n"
        "print('Selected blend weights:', best_w)\n\n"
        "# Task B threshold tuning\n"
        "b_s2_all = np.vstack([wB * s2_mean_c[1] + (1-wB)*tf_s2_pbc, wB * s2_mean_n[1] + (1-wB)*tf_s2_pbn])\n"
        "y_s2_b = np.vstack([real_tech, real_tech])\n"
        "b_s1_all = np.vstack([wB * x1_c[1] + (1-wB)*tf_s1_pbc, wB * x1_n[1] + (1-wB)*tf_s1_pbn])\n"
        "y_s1_b = np.vstack([synth_tech[va_f0], synth_tech[va_f0]])\n"
        "b_ho = np.vstack([b_s2_all, b_s1_all])\n"
        "y_ho_b = np.vstack([y_s2_b, y_s1_b])\n"
        "shrunk_t = np.full(6, 0.5)\n"
        "for j in range(6):\n"
        "    bt = max(np.round(np.arange(0.30, 0.71, 0.05), 2), key=lambda t: f1_score(y_ho_b[:, j], (b_ho[:, j] >= t).astype(int), zero_division=0))\n"
        "    shrunk_t[j] = 0.5 + 0.5 * (bt - 0.5)\n"
        "s2_before = np.mean([f1_score(y_s2_b[:, j], (b_s2_all[:, j]>=0.5).astype(int), zero_division=0) for j in range(6)])\n"
        "s2_after = np.mean([f1_score(y_s2_b[:, j], (b_s2_all[:, j]>=shrunk_t[j]).astype(int), zero_division=0) for j in range(6)])\n"
        "s1_before = np.mean([f1_score(y_s1_b[:, j], (b_s1_all[:, j]>=0.5).astype(int), zero_division=0) for j in range(6)])\n"
        "s1_after = np.mean([f1_score(y_s1_b[:, j], (b_s1_all[:, j]>=shrunk_t[j]).astype(int), zero_division=0) for j in range(6)])\n"
        "FINAL_TECH_THRESH = shrunk_t if ((s2_after - s2_before > 0.01) and (s1_after - s1_before > 0.01)) else np.full(6, 0.5)\n"
        "print('FINAL_TECH_THRESH:', FINAL_TECH_THRESH)\n\n"
        "# Confidence calibration\n"
        "ho_pa = np.vstack([wA*s2_mean_c[0]+(1-wA)*tf_s2_pac, wA*s2_mean_n[0]+(1-wA)*tf_s2_pan, wA*x1_c[0]+(1-wA)*tf_s1_pac, wA*x1_n[0]+(1-wA)*tf_s1_pan])\n"
        "ho_y_a = np.concatenate([real_lbl, real_lbl, synth_lbl[va_f0], synth_lbl[va_f0]])\n"
        "ho_corr = (np.argmax(ho_pa, 1) == ho_y_a).astype(int)\n"
        "ho_raw = np.max(ho_pa, 1)\n"
        "log_ho = np.log(np.clip(ho_pa, 1e-12, 1.0))\n"
        "opt_T = float(minimize_scalar(lambda T: -np.mean(np.log(np.clip((np.exp(log_ho/T - np.max(log_ho/T,1,keepdims=True)) / np.sum(np.exp(log_ho/T - np.max(log_ho/T,1,keepdims=True)),1,keepdims=True))[np.arange(len(ho_y_a)), ho_y_a], 1e-12, 1.0))), bounds=(0.2, 5.0), method='bounded').x)\n"
        "def apply_temp(probs, T):\n"
        "    s = np.log(np.clip(probs, 1e-12, 1.0)) / T; s -= np.max(s, 1, keepdims=True); ex = np.exp(s)\n"
        "    return np.max(ex / np.sum(ex, 1, keepdims=True), 1)\n"
        "iso_cal = IsotonicRegression(y_min=0.05, y_max=0.98, out_of_bounds='clip').fit(ho_raw, ho_corr)\n"
        "platt_cal = LogisticRegression(C=1.0, max_iter=500, random_state=SEED).fit(ho_raw.reshape(-1,1), ho_corr)\n"
        "cal_scores = {'raw': brier_score_loss(ho_corr, ho_raw), 'temperature': brier_score_loss(ho_corr, apply_temp(ho_pa, opt_T)), 'isotonic': brier_score_loss(ho_corr, iso_cal.predict(ho_raw)), 'platt': brier_score_loss(ho_corr, platt_cal.predict_proba(ho_raw.reshape(-1,1))[:,1])}\n"
        "best_cal = min(cal_scores, key=cal_scores.get)\n"
        "print('Calibrator:', best_cal, cal_scores)\n"
        "def calibrate_conf(probs_a):\n"
        "    rm = np.max(probs_a, 1)\n"
        "    c = rm if best_cal=='raw' else (apply_temp(probs_a, opt_T) if best_cal=='temperature' else (iso_cal.predict(rm) if best_cal=='isotonic' else platt_cal.predict_proba(rm.reshape(-1,1))[:,1]))\n"
        "    return np.round(np.clip(c, 0.05, 0.98), 4)\n"
    ))

    # 7. Final v3 Training (2 XLM-R seeds + TF-IDF), 500k Inference, Post-Processing & Validator
    cells.append(nbf.v4.new_markdown_cell("## 4. Final `synthetic_v3` Training (XLM-R Seed 42 & Seed 7 + TF-IDF), 500k Inference & Validation"))
    cells.append(nbf.v4.new_code_cell(
        "def generate_noise_only_rows(n_rows=300, seed=SEED):\n"
        "    rng_n = random.Random(seed + 999)\n"
        "    cons, lets = 'bcdfghjklmnpqrstvwxyz', 'abcdefghijklmnopqrstuvwxyz'\n"
        "    rows = []\n"
        "    for _ in range(n_rows):\n"
        "        toks = [''.join(rng_n.choice(cons if rng_n.random() < 0.65 else lets) for _ in range(rng_n.randint(5, 13))) for _ in range(rng_n.randint(1, 3))]\n"
        "        txt = ' '.join(toks)\n"
        "        if rng_n.random() < 0.25: txt += rng_n.choice(['', '.', '!', '?', '...'])\n"
        "        if rng_n.random() < 0.20:\n"
        "            em = rng_n.choice(['📲', '⚠️', '🔥', '💰', ''])\n"
        "            if em: txt = (em + ' ' + txt) if rng_n.random() < 0.5 else (txt + ' ' + em)\n"
        "        rows.append(preprocess_text(txt))\n"
        "    return rows\n\n"
        "noise_only_texts = generate_noise_only_rows(300, seed=SEED)\n"
        "n_noise = len(noise_only_texts)\n"
        "v3_lbl, v3_tech, v3_act = v3_df['label'].astype(int).values, encode_techniques(v3_df['techniques']), v3_df['action'].astype(int).values\n"
        "final_v3_texts = v3_df['clean_text'].tolist() + noise_only_texts + real_df['clean_text'].tolist() + real_df['noisy_text'].tolist()\n"
        "final_v3_lbl = np.concatenate([v3_lbl, np.zeros(n_noise, int), real_lbl, real_lbl])\n"
        "final_v3_tech = np.vstack([v3_tech, np.zeros((n_noise, 6), int), real_tech, real_tech])\n"
        "final_v3_act = np.concatenate([v3_act, np.zeros(n_noise, int), real_act, real_act])\n"
        "final_v3_weights = np.concatenate([np.ones(len(v3_df), np.float32), np.ones(n_noise, np.float32), np.full(len(real_df), 5.0, np.float32), np.full(len(real_df), 5.0, np.float32)])\n\n"
        "TRAIN_ALPHA_VOCAB_V3 = {tok.strip('.,!?:;\"\\'()[]{}') for txt in final_v3_texts for tok in txt.split() if tok not in ('[URL]', '[NUM]', '|||') and any(c.isalpha() for c in tok)}\n\n"
        "seed_everything(42)\n"
        "xlmr_v3_s42_ckpt = os.path.join(WORK_DIR, 'xlmr_final_v3_s42.pt')\n"
        "m_v3_s42, tok_v3_s42 = train_three_head('XLM-R v3 (seed=42)', final_v3_texts, final_v3_lbl, final_v3_tech, final_v3_act, final_v3_weights, xlmr_v3_s42_ckpt, BACKBONE)\n"
        "free_gpu('m_v3_s42')\n\n"
        "seed_everything(7)\n"
        "xlmr_v3_s7_ckpt = os.path.join(WORK_DIR, 'xlmr_final_v3_s7.pt')\n"
        "m_v3_s7, tok_v3_s7 = train_three_head('XLM-R v3 (seed=7)', final_v3_texts, final_v3_lbl, final_v3_tech, final_v3_act, final_v3_weights, xlmr_v3_s7_ckpt, BACKBONE)\n"
        "free_gpu('m_v3_s7')\n"
        "seed_everything(42)\n\n"
        "v3_cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, dtype=np.float32)\n"
        "v3_wv = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)\n"
        "X_v3 = hstack([v3_cv.fit_transform(final_v3_texts), v3_wv.fit_transform(final_v3_texts)]).tocsr()\n"
        "v3_ca = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED).fit(X_v3, final_v3_lbl, sample_weight=final_v3_weights)\n"
        "v3_cb = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)).fit(X_v3, final_v3_tech)\n"
        "v3_cc = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED).fit(X_v3, final_v3_act, sample_weight=final_v3_weights)\n\n"
        "test_df = pd.read_csv(os.path.join(COMP_DIR, 'test.csv'))\n"
        "N_TEST, CHUNK_SIZE = len(test_df), 50000\n"
        "def run_xlmr_chunks(ckpt_path, tok, prefix):\n"
        "    pa_l, pb_l, pc_l, mdl = [], [], [], None\n"
        "    for start in range(0, N_TEST, CHUNK_SIZE):\n"
        "        end, c_idx = min(start + CHUNK_SIZE, N_TEST), start // CHUNK_SIZE\n"
        "        c_file = os.path.join(WORK_DIR, f'{prefix}_{c_idx}.npz')\n"
        "        if os.path.isfile(c_file):\n"
        "            d = np.load(c_file); pa_l.append(d['pa']); pb_l.append(d['pb']); pc_l.append(d['pc']); continue\n"
        "        if mdl is None:\n"
        "            mdl = ThreeHeadEncoder(BACKBONE).to(DEVICE); mdl.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)); mdl.eval()\n"
        "        raw_s = test_df['text'].iloc[start:end].fillna('').astype(str).apply(preprocess_text).tolist()\n"
        "        order = np.argsort([len(t) for t in raw_s])\n"
        "        pa_s, pb_s, pc_s = predict_three_head(mdl, [raw_s[i] for i in order], tok, batch_size=256, max_len=256)\n"
        "        inv = np.argsort(order)\n"
        "        pa_c, pb_c, pc_c = pa_s[inv], pb_s[inv], pc_s[inv]\n"
        "        np.savez(c_file, pa=pa_c, pb=pb_c, pc=pc_c)\n"
        "        pa_l.append(pa_c); pb_l.append(pb_c); pc_l.append(pc_c)\n"
        "    if mdl is not None:\n"
        "        del mdl; gc.collect()\n"
        "        if torch.cuda.is_available(): torch.cuda.empty_cache()\n"
        "    return np.vstack(pa_l), np.vstack(pb_l), np.vstack(pc_l)\n\n"
        "v3_s42_pa, v3_s42_pb, v3_s42_pc = run_xlmr_chunks(xlmr_v3_s42_ckpt, tok_v3_s42, 'test_xlmr_v3_s42_chunk')\n"
        "v3_s7_pa, v3_s7_pb, v3_s7_pc = run_xlmr_chunks(xlmr_v3_s7_ckpt, tok_v3_s7, 'test_xlmr_v3_s7_chunk')\n\n"
        "clean_all_test = test_df['text'].fillna('').astype(str).apply(preprocess_text).tolist()\n"
        "tf_pa_l, tf_pb_l, tf_pc_l = [], [], []\n"
        "for start in range(0, N_TEST, CHUNK_SIZE):\n"
        "    end = min(start + CHUNK_SIZE, N_TEST)\n"
        "    X_t = hstack([v3_cv.transform(clean_all_test[start:end]), v3_wv.transform(clean_all_test[start:end])]).tocsr()\n"
        "    tf_pa_l.append(v3_ca.predict_proba(X_t).astype(np.float32)); tf_pb_l.append(v3_cb.predict_proba(X_t).astype(np.float32)); tf_pc_l.append(v3_cc.predict_proba(X_t).astype(np.float32))\n"
        "v3_pa = wA * (0.5 * (v3_s42_pa + v3_s7_pa)) + (1-wA) * np.vstack(tf_pa_l)\n"
        "v3_pb = wB * (0.5 * (v3_s42_pb + v3_s7_pb)) + (1-wB) * np.vstack(tf_pb_l)\n"
        "v3_pc = wC * (0.5 * (v3_s42_pc + v3_s7_pc)) + (1-wC) * np.vstack(tf_pc_l)\n\n"
        "v3_labels, v3_actions, v3_confs, v3_techs = np.argmax(v3_pa, 1), np.argmax(v3_pc, 1), calibrate_conf(v3_pa), []\n"
        "def is_oov_noise_v3(txt):\n"
        "    if '[URL]' in txt or '[NUM]' in txt: return False\n"
        "    at = [tok.strip('.,!?:;\"\\'()[]{}') for tok in txt.split() if tok != '|||' and any(c.isalpha() for c in tok)]\n"
        "    return bool(at) and all(t not in TRAIN_ALPHA_VOCAB_V3 for t in at)\n"
        "for i in range(N_TEST):\n"
        "    if is_oov_noise_v3(clean_all_test[i]):\n"
        "        v3_labels[i] = 0; v3_techs.append('NONE'); v3_actions[i] = 0; v3_confs[i] = 0.5; continue\n"
        "    if v3_labels[i] in (0, 5):\n"
        "        v3_techs.append('NONE'); v3_actions[i] = 0\n"
        "    else:\n"
        "        act_t = [TECH_CODES[j] for j in range(6) if v3_pb[i, j] >= FINAL_TECH_THRESH[j]]\n"
        "        if not act_t: act_t = [TECH_CODES[int(np.argmax(v3_pb[i]))]]\n"
        "        v3_techs.append('|'.join(sorted(act_t)))\n\n"
        "sub_out = os.path.join(WORK_DIR, 'submission.csv')\n"
        "pd.DataFrame({'id': test_df['id'].values.astype(int), 'label': v3_labels.astype(int), 'techniques': v3_techs, 'action': v3_actions.astype(int), 'confidence': v3_confs}).to_csv(sub_out, index=False)\n"
        "ref_ids = pd.read_csv(os.path.join(COMP_DIR, 'sample_submission.csv'), usecols=['id'])['id'].values\n"
        "chk = pd.read_csv(sub_out, keep_default_na=False)\n"
        "assert list(chk.columns) == ['id', 'label', 'techniques', 'action', 'confidence'] and len(chk) == N_TEST == len(ref_ids)\n"
        "assert (chk['id'].astype(int).values == test_df['id'].values).all() and not chk['id'].duplicated().any()\n"
        "assert chk['label'].astype(int).isin([0,1,2,3,4,5]).all() and chk['action'].astype(int).isin([0,1,2,3,4,5]).all()\n"
        "print('PASS: submission.csv generated & validated! Rows:', len(chk))\n"
    ))

    nb["cells"] = cells
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse(c["source"])

    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Built final self-contained notebook.ipynb ({len(cells)} cells, all code cells syntax-checked).")


if __name__ == "__main__":
    build_final_notebook()
