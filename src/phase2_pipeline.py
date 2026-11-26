import os
import random
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix

from preprocess import preprocess_text
from synth import generate_synthetic_dataset
from noise import apply_noise
from validate_submission import validate_submission, get_data_dir

SEED = 42
TECH_CODES = ["T1", "T2", "T3", "T4", "T5", "T6"]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def encode_techniques(series: pd.Series) -> np.ndarray:
    y = np.zeros((len(series), len(TECH_CODES)), dtype=int)
    for i, val in enumerate(series.astype(str)):
        parts = val.split("|")
        for j, code in enumerate(TECH_CODES):
            if code in parts:
                y[i, j] = 1
    return y


def postprocess_predictions(labels, tech_probs, actions, confs):
    out_techs = []
    out_actions = actions.copy()
    for i in range(len(labels)):
        if labels[i] == 0:
            out_techs.append("NONE")
            out_actions[i] = 0
        else:
            active = [TECH_CODES[j] for j in range(len(TECH_CODES)) if tech_probs[i, j] >= 0.5]
            if not active:
                active = [TECH_CODES[int(np.argmax(tech_probs[i]))]]
            out_techs.append("|".join(sorted(active)))
    out_confs = np.round(np.clip(confs, 0.05, 0.98), 4)
    return labels, out_techs, out_actions, out_confs


def fit_vectorizers(texts):
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, dtype=np.float32
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32
    )
    X_char = char_vec.fit_transform(texts)
    X_word = word_vec.fit_transform(texts)
    return char_vec, word_vec, hstack([X_char, X_word]).tocsr()


def transform_vectorizers(char_vec, word_vec, texts):
    return hstack([char_vec.transform(texts), word_vec.transform(texts)]).tocsr()


def fit_and_predict(tr_texts, tr_lbl, tr_tech, tr_act, va_texts, sample_weight=None):
    cv_c, cv_w, X_tr = fit_vectorizers(tr_texts)
    X_va = transform_vectorizers(cv_c, cv_w, va_texts)

    ca = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    ca.fit(X_tr, tr_lbl, sample_weight=sample_weight)
    pa = ca.predict_proba(X_va)
    pred_lbl = np.argmax(pa, axis=1)
    pred_conf = np.max(pa, axis=1)

    cb = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    )
    cb.fit(X_tr, tr_tech)
    prob_tech = cb.predict_proba(X_va)

    cc = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    cc.fit(X_tr, tr_act, sample_weight=sample_weight)
    pred_act = cc.predict(X_va)

    return postprocess_predictions(pred_lbl, prob_tech, pred_act, pred_conf)


def compute_metrics(y_lbl, y_tech, y_act, p_lbl, p_tech_strs, p_act):
    p_tech_bin = encode_techniques(pd.Series(p_tech_strs))
    f1_a = f1_score(y_lbl, p_lbl, average="macro")
    f1_b = float(np.mean([f1_score(y_tech[:, j], p_tech_bin[:, j], zero_division=0) for j in range(6)]))
    f1_c = f1_score(y_act, p_act, average="macro")
    return f1_a, f1_b, f1_c


def run_phase2():
    set_seed(SEED)
    rng = random.Random(SEED)
    t0 = time.time()
    data_dir = get_data_dir()

    # 1. Load real sample.csv & build noisy copies
    real_df = pd.read_csv(os.path.join(data_dir, "sample.csv"))
    real_clean_texts = real_df["text"].fillna("").astype(str).apply(preprocess_text).tolist()
    real_noisy_raw = [apply_noise(t, rng, level=0.35) for t in real_df["text"].fillna("").astype(str)]
    real_noisy_texts = [preprocess_text(t) for t in real_noisy_raw]

    real_lbl = real_df["label"].astype(int).values
    real_tech = encode_techniques(real_df["techniques"])
    real_act = real_df["action"].astype(int).values

    # 2. Generate synthetic dataset + apply noise to ~30% of rows
    synth_df = generate_synthetic_dataset(target_rows=6000, seed=SEED)
    is_noisy_flags = []
    aug_texts = []
    for txt in synth_df["text"].tolist():
        if rng.random() < 0.30:
            aug_texts.append(apply_noise(txt, rng, level=0.35))
            is_noisy_flags.append(True)
        else:
            aug_texts.append(txt)
            is_noisy_flags.append(False)

    synth_df["text"] = aug_texts
    synth_df["is_noisy"] = is_noisy_flags

    os.makedirs("data", exist_ok=True)
    synth_df.to_csv("data/synthetic.csv", index=False)
    print(f"Saved data/synthetic.csv: shape={synth_df.shape}, noisy={synth_df['is_noisy'].mean()*100:.1f}%")

    synth_clean_proc = synth_df["text"].apply(preprocess_text).tolist()
    synth_lbl = synth_df["label"].astype(int).values
    synth_tech = encode_techniques(synth_df["techniques"])
    synth_act = synth_df["action"].astype(int).values
    synth_groups = synth_df["family_id"].values

    # ==================== STEP 4: HONEST VALIDATION ====================
    print("\n" + "=" * 70)
    print("STEP 4: HONEST VALIDATION (CLEAN vs NOISY)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # (a) Phase 1 baseline (86 real rows only) on 5-fold CV
    a_cln_lbl, a_cln_tech, a_cln_act = np.zeros(len(real_df), int), [""] * len(real_df), np.zeros(len(real_df), int)
    a_nsy_lbl, a_nsy_tech, a_nsy_act = np.zeros(len(real_df), int), [""] * len(real_df), np.zeros(len(real_df), int)
    for tr_i, va_i in skf.split(real_clean_texts, real_lbl):
        pl, pt, pa, _ = fit_and_predict(
            [real_clean_texts[i] for i in tr_i], real_lbl[tr_i], real_tech[tr_i], real_act[tr_i],
            [real_clean_texts[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            a_cln_lbl[idx], a_cln_tech[idx], a_cln_act[idx] = pl[k], pt[k], pa[k]

        pl_n, pt_n, pa_n, _ = fit_and_predict(
            [real_clean_texts[i] for i in tr_i], real_lbl[tr_i], real_tech[tr_i], real_act[tr_i],
            [real_noisy_texts[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            a_nsy_lbl[idx], a_nsy_tech[idx], a_nsy_act[idx] = pl_n[k], pt_n[k], pa_n[k]

    fa_a_c, fb_a_c, fc_a_c = compute_metrics(real_lbl, real_tech, real_act, a_cln_lbl, a_cln_tech, a_cln_act)
    fa_a_n, fb_a_n, fc_a_n = compute_metrics(real_lbl, real_tech, real_act, a_nsy_lbl, a_nsy_tech, a_nsy_act)
    print(f"(a) Real-only 5-fold CV  | CLEAN: A={fa_a_c:.4f}, B={fb_a_c:.4f}, C={fc_a_c:.4f} | NOISY: A={fa_a_n:.4f}, B={fb_a_n:.4f}, C={fc_a_n:.4f}")

    # (b) Train on synthetic only -> evaluate on all 86 real rows
    b_cln_lbl, b_cln_tech, b_cln_act, _ = fit_and_predict(synth_clean_proc, synth_lbl, synth_tech, synth_act, real_clean_texts)
    b_nsy_lbl, b_nsy_tech, b_nsy_act, _ = fit_and_predict(synth_clean_proc, synth_lbl, synth_tech, synth_act, real_noisy_texts)
    fa_b_c, fb_b_c, fc_b_c = compute_metrics(real_lbl, real_tech, real_act, b_cln_lbl, b_cln_tech, b_cln_act)
    fa_b_n, fb_b_n, fc_b_n = compute_metrics(real_lbl, real_tech, real_act, b_nsy_lbl, b_nsy_tech, b_nsy_act)
    print(f"(b) Synth-only -> Real   | CLEAN: A={fa_b_c:.4f}, B={fb_b_c:.4f}, C={fc_b_c:.4f} | NOISY: A={fa_b_n:.4f}, B={fb_b_n:.4f}, C={fc_b_n:.4f}")

    # (c) 5-fold CV on real rows + ALL synthetic in train fold (real weight 5x)
    c_cln_lbl, c_cln_tech, c_cln_act = np.zeros(len(real_df), int), [""] * len(real_df), np.zeros(len(real_df), int)
    c_nsy_lbl, c_nsy_tech, c_nsy_act = np.zeros(len(real_df), int), [""] * len(real_df), np.zeros(len(real_df), int)
    for tr_i, va_i in skf.split(real_clean_texts, real_lbl):
        tr_texts = [real_clean_texts[i] for i in tr_i] + synth_clean_proc
        tr_lbl = np.concatenate([real_lbl[tr_i], synth_lbl])
        tr_tech = np.vstack([real_tech[tr_i], synth_tech])
        tr_act = np.concatenate([real_act[tr_i], synth_act])
        weights = np.concatenate([np.full(len(tr_i), 5.0), np.ones(len(synth_lbl))])

        pl, pt, pa, _ = fit_and_predict(tr_texts, tr_lbl, tr_tech, tr_act, [real_clean_texts[i] for i in va_i], sample_weight=weights)
        for k, idx in enumerate(va_i):
            c_cln_lbl[idx], c_cln_tech[idx], c_cln_act[idx] = pl[k], pt[k], pa[k]

        pl_n, pt_n, pa_n, _ = fit_and_predict(tr_texts, tr_lbl, tr_tech, tr_act, [real_noisy_texts[i] for i in va_i], sample_weight=weights)
        for k, idx in enumerate(va_i):
            c_nsy_lbl[idx], c_nsy_tech[idx], c_nsy_act[idx] = pl_n[k], pt_n[k], pa_n[k]

    fa_c_c, fb_c_c, fc_c_c = compute_metrics(real_lbl, real_tech, real_act, c_cln_lbl, c_cln_tech, c_cln_act)
    fa_c_n, fb_c_n, fc_c_n = compute_metrics(real_lbl, real_tech, real_act, c_nsy_lbl, c_nsy_tech, c_nsy_act)
    print(f"(c) Real(5x)+Synth CV    | CLEAN: A={fa_c_c:.4f}, B={fb_c_c:.4f}, C={fc_c_c:.4f} | NOISY: A={fa_c_n:.4f}, B={fb_c_n:.4f}, C={fc_c_n:.4f}")

    # Per-class F1 & Confusion Matrix for (c) Clean
    per_cls_c = f1_score(real_lbl, c_cln_lbl, average=None)
    print("\nPer-class F1 for (c) CLEAN:", {k: round(float(per_cls_c[k]), 4) for k in range(6)})
    cm_c = confusion_matrix(real_lbl, c_cln_lbl)
    print("Confusion Matrix for (c) CLEAN (rows=true, cols=pred):\n", cm_c)
    per_cls_n = f1_score(real_lbl, c_nsy_lbl, average=None)
    print("Per-class F1 for (c) NOISY:", {k: round(float(per_cls_n[k]), 4) for k in range(6)})

    # (d) StratifiedGroupKFold on synthetic data grouped by family_id
    sgkf = StratifiedGroupKFold(n_splits=5)
    d_cln_lbl, d_cln_tech, d_cln_act = np.zeros(len(synth_df), int), [""] * len(synth_df), np.zeros(len(synth_df), int)
    d_nsy_lbl, d_nsy_tech, d_nsy_act = np.zeros(len(synth_df), int), [""] * len(synth_df), np.zeros(len(synth_df), int)
    synth_all_noisy_proc = [preprocess_text(apply_noise(t, rng, level=0.35)) for t in synth_df["text"].tolist()]

    for tr_i, va_i in sgkf.split(synth_clean_proc, synth_lbl, synth_groups):
        pl, pt, pa, _ = fit_and_predict(
            [synth_clean_proc[i] for i in tr_i], synth_lbl[tr_i], synth_tech[tr_i], synth_act[tr_i],
            [synth_clean_proc[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            d_cln_lbl[idx], d_cln_tech[idx], d_cln_act[idx] = pl[k], pt[k], pa[k]

        pl_n, pt_n, pa_n, _ = fit_and_predict(
            [synth_clean_proc[i] for i in tr_i], synth_lbl[tr_i], synth_tech[tr_i], synth_act[tr_i],
            [synth_all_noisy_proc[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            d_nsy_lbl[idx], d_nsy_tech[idx], d_nsy_act[idx] = pl_n[k], pt_n[k], pa_n[k]

    fa_d_c, fb_d_c, fc_d_c = compute_metrics(synth_lbl, synth_tech, synth_act, d_cln_lbl, d_cln_tech, d_cln_act)
    fa_d_n, fb_d_n, fc_d_n = compute_metrics(synth_lbl, synth_tech, synth_act, d_nsy_lbl, d_nsy_tech, d_nsy_act)
    print(f"\n(d) Synth GroupKFold     | CLEAN: A={fa_d_c:.4f}, B={fb_d_c:.4f}, C={fc_d_c:.4f} | NOISY: A={fa_d_n:.4f}, B={fb_d_n:.4f}, C={fc_d_n:.4f}")

    # ==================== STEP 5: RETRAIN & PREDICT ====================
    print("\n" + "=" * 70)
    print("STEP 5: RETRAIN PHASE 2 MODEL & PREDICT test.csv")
    print("=" * 70)
    full_texts = real_clean_texts + real_noisy_texts + synth_clean_proc
    full_lbl = np.concatenate([real_lbl, real_lbl, synth_lbl])
    full_tech = np.vstack([real_tech, real_tech, synth_tech])
    full_act = np.concatenate([real_act, real_act, synth_act])
    full_weights = np.concatenate([
        np.full(len(real_lbl), 5.0),
        np.full(len(real_lbl), 3.0),
        np.ones(len(synth_lbl)),
    ])

    char_vec, word_vec, X_full = fit_vectorizers(full_texts)
    full_clf_a = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_full, full_lbl, sample_weight=full_weights)
    full_clf_b = OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)).fit(X_full, full_tech)
    full_clf_c = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_full, full_act, sample_weight=full_weights)

    test_path = os.path.join(data_dir, "test.csv")
    out_path = "submission_phase2.csv"
    first_chunk = True
    label_counts = {k: 0 for k in range(6)}
    total_rows = 0

    for chunk in pd.read_csv(test_path, chunksize=50000):
        clean_batch = chunk["text"].fillna("").astype(str).apply(preprocess_text).tolist()
        X_test = transform_vectorizers(char_vec, word_vec, clean_batch)
        probs_a = full_clf_a.predict_proba(X_test)
        preds_a = np.argmax(probs_a, axis=1)
        confs = np.max(probs_a, axis=1)
        tech_probs = full_clf_b.predict_proba(X_test)
        preds_c = full_clf_c.predict(X_test)

        preds_a, tech_strs, preds_c, confs = postprocess_predictions(preds_a, tech_probs, preds_c, confs)
        for lbl in preds_a:
            label_counts[int(lbl)] += 1

        pd.DataFrame({
            "id": chunk["id"].values.astype(int),
            "label": preds_a.astype(int),
            "techniques": tech_strs,
            "action": preds_c.astype(int),
            "confidence": confs,
        }).to_csv(out_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(chunk)

    print(f"Saved {out_path} ({total_rows:,} rows) in {time.time() - t0:.2f}s")
    print("Phase 2 Predicted Label Distribution on test.csv:")
    for k, v in label_counts.items():
        print(f"  Label {k}: {v:7,d} ({v/total_rows*100:5.2f}%)")

    print("\nValidating submission_phase2.csv:")
    validate_submission(out_path)

    # ==================== STEP 6: QUALITY CHECK (40 SAMPLES) ====================
    print("\n" + "=" * 70)
    print("STEP 6: QUALITY CHECK (40 GENERATED ROWS)")
    print("=" * 70)
    sampled_idx = set()
    for lang in ["english", "bangla", "banglish", "mixed"]:
        sampled_idx.update(synth_df[synth_df["language"] == lang].sample(5, random_state=SEED).index)
    for lbl in range(6):
        sampled_idx.update(synth_df[synth_df["label"] == lbl].sample(3, random_state=SEED).index)
    sampled_idx.update(synth_df[synth_df["n_turns"] > 1].sample(5, random_state=SEED).index)
    sampled_idx.update(synth_df[synth_df["is_hard_negative"]].sample(5, random_state=SEED).index)
    sampled_idx.update(synth_df[synth_df["is_noisy"]].sample(5, random_state=SEED).index)

    qc_list = sorted(list(sampled_idx))[:40]
    for i, idx in enumerate(qc_list):
        r = synth_df.loc[idx]
        print(f"[{i+1:02d}] L={r['label']} | T={r['techniques']:8s} | A={r['action']} | {r['language']:8s} | turns={r['n_turns']} | hard={r['is_hard_negative']} | noisy={r['is_noisy']}")
        print(f"     {repr(r['text'][:140])}")


if __name__ == "__main__":
    run_phase2()
