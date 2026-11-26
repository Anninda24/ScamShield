import os
import random
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from preprocess import preprocess_text
from noise import apply_noise
from phase2_pipeline import (
    encode_techniques,
    postprocess_predictions,
    fit_vectorizers,
    transform_vectorizers,
    fit_and_predict,
    compute_metrics,
    SEED,
)
from validate_submission import validate_submission


def run_phase3a_check():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)
    t0 = time.time()

    real_df = pd.read_csv("sample.csv")
    v2_df = pd.read_csv("data/synthetic_v2.csv", keep_default_na=False)

    real_clean = real_df["text"].fillna("").astype(str).apply(preprocess_text).tolist()
    real_noisy = [preprocess_text(apply_noise(t, rng, level=0.35)) for t in real_df["text"].fillna("").astype(str)]
    real_lbl = real_df["label"].astype(int).values
    real_tech = encode_techniques(real_df["techniques"])
    real_act = real_df["action"].astype(int).values
    real_ids = real_df["id"].astype(int).values

    # ==================== 1A. REAL-ROW CHECK: synthetic_v2 ONLY ====================
    print("=" * 70)
    print("1A. REAL-ROW CHECK: TRAIN ON synthetic_v2 ONLY (NO real_paraphrase, NO synthetic_v1)")
    print("=" * 70)
    s2_only = v2_df[v2_df["source"] == "synthetic_v2"].reset_index(drop=True)
    s2_proc = s2_only["text"].astype(str).apply(preprocess_text).tolist()
    s2_lbl = s2_only["label"].astype(int).values
    s2_tech = encode_techniques(s2_only["techniques"])
    s2_act = s2_only["action"].astype(int).values
    print(f"Training rows (source=='synthetic_v2'): {len(s2_only)}")

    pl_c, pt_c, pa_c, _ = fit_and_predict(s2_proc, s2_lbl, s2_tech, s2_act, real_clean)
    pl_n, pt_n, pa_n, _ = fit_and_predict(s2_proc, s2_lbl, s2_tech, s2_act, real_noisy)

    fa_c, fb_c, fc_c = compute_metrics(real_lbl, real_tech, real_act, pl_c, pt_c, pa_c)
    fa_n, fb_n, fc_n = compute_metrics(real_lbl, real_tech, real_act, pl_n, pt_n, pa_n)
    print(f"CLEAN 86 Real: Task A={fa_c:.4f}, Task B={fb_c:.4f}, Task C={fc_c:.4f}")
    print("Confusion Matrix (CLEAN Task A):\n", confusion_matrix(real_lbl, pl_c))
    print(f"NOISY 86 Real: Task A={fa_n:.4f}, Task B={fb_n:.4f}, Task C={fc_n:.4f}")
    print("Confusion Matrix (NOISY Task A):\n", confusion_matrix(real_lbl, pl_n))

    # ==================== 1B. 5-FOLD OVER REAL ROWS WITH NON-LEAKING PARAPHRASES ====================
    print("\n" + "=" * 70)
    print("1B. 5-FOLD OVER REAL ROWS: synthetic_v2 + TRAIN-FOLD real_paraphrase ONLY")
    print("=" * 70)
    para_df = v2_df[v2_df["source"] == "real_paraphrase"].copy()
    para_df["parent_id"] = para_df["scenario_id"].str.replace("real_para_", "", regex=False).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_c_lbl, oof_c_tech, oof_c_act = np.zeros(len(real_df), int), [""] * len(real_df), np.zeros(len(real_df), int)
    oof_n_lbl, oof_n_tech, oof_n_act = np.zeros(len(real_df), int), [""] * len(real_df), np.zeros(len(real_df), int)

    for tr_i, va_i in skf.split(real_clean, real_lbl):
        tr_parent_ids = set(real_ids[tr_i])
        fold_para = para_df[para_df["parent_id"].isin(tr_parent_ids)]
        fp_proc = fold_para["text"].astype(str).apply(preprocess_text).tolist()
        fp_lbl = fold_para["label"].astype(int).values
        fp_tech = encode_techniques(fold_para["techniques"])
        fp_act = fold_para["action"].astype(int).values

        tr_texts = s2_proc + fp_proc
        tr_lbl = np.concatenate([s2_lbl, fp_lbl])
        tr_tech = np.vstack([s2_tech, fp_tech])
        tr_act = np.concatenate([s2_act, fp_act])

        plc, ptc, pac, _ = fit_and_predict(tr_texts, tr_lbl, tr_tech, tr_act, [real_clean[i] for i in va_i])
        pln, ptn, pan, _ = fit_and_predict(tr_texts, tr_lbl, tr_tech, tr_act, [real_noisy[i] for i in va_i])
        for k, idx in enumerate(va_i):
            oof_c_lbl[idx], oof_c_tech[idx], oof_c_act[idx] = plc[k], ptc[k], pac[k]
            oof_n_lbl[idx], oof_n_tech[idx], oof_n_act[idx] = pln[k], ptn[k], pan[k]

    fa_1b_c, fb_1b_c, fc_1b_c = compute_metrics(real_lbl, real_tech, real_act, oof_c_lbl, oof_c_tech, oof_c_act)
    fa_1b_n, fb_1b_n, fc_1b_n = compute_metrics(real_lbl, real_tech, real_act, oof_n_lbl, oof_n_tech, oof_n_act)
    print(f"CLEAN 86 Real (5-fold w/ train paraphrases): Task A={fa_1b_c:.4f}, Task B={fb_1b_c:.4f}, Task C={fc_1b_c:.4f}")
    print("Confusion Matrix (CLEAN Task A):\n", confusion_matrix(real_lbl, oof_c_lbl))
    print(f"NOISY 86 Real (5-fold w/ train paraphrases): Task A={fa_1b_n:.4f}, Task B={fb_1b_n:.4f}, Task C={fc_1b_n:.4f}")
    print("Confusion Matrix (NOISY Task A):\n", confusion_matrix(real_lbl, oof_n_lbl))

    # ==================== 2. CROSS-SCENARIO LEAKAGE ====================
    print("\n" + "=" * 70)
    print("2. CROSS-SCENARIO LEAKAGE (NEAREST NEIGHBOR FROM DIFFERENT scenario_id)")
    print("=" * 70)
    # Check across all synthetic rows (v1 + v2)
    cs_df = v2_df[v2_df["source"] != "real_paraphrase"].reset_index(drop=True)
    cs_proc = cs_df["text"].astype(str).apply(preprocess_text).tolist()
    cs_scens = cs_df["scenario_id"].values
    cs_lbls = cs_df["label"].astype(int).values

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    X_cs = normalize(char_vec.fit_transform(cs_proc))

    n_cs = len(cs_df)
    nn_sims = np.zeros(n_cs, dtype=np.float32)
    nn_idx = np.zeros(n_cs, dtype=int)

    # Batched cosine similarity to stay fast & low memory
    batch_sz = 1000
    for start in range(0, n_cs, batch_sz):
        end = min(start + batch_sz, n_cs)
        sims = (X_cs[start:end] @ X_cs.T).toarray()
        # Mask out same scenario_id
        for r in range(end - start):
            sims[r, cs_scens == cs_scens[start + r]] = -1.0
        best_j = np.argmax(sims, axis=1)
        nn_idx[start:end] = best_j
        nn_sims[start:end] = sims[np.arange(end - start), best_j]

    same_lbl_pct = (cs_lbls[nn_idx] == cs_lbls).mean() * 100
    print(f"Rows evaluated (synthetic_v1 + synthetic_v2): {n_cs}")
    print(f"Cross-scenario max similarity:")
    print(f"  Mean   : {nn_sims.mean():.4f}")
    print(f"  Median : {np.median(nn_sims):.4f}")
    print(f"  >= 0.8 : {(nn_sims >= 0.8).mean() * 100:.2f}% ({(nn_sims >= 0.8).sum()}/{n_cs})")
    print(f"  Nearest neighbor has SAME label: {same_lbl_pct:.2f}%")

    # Top 10 closest cross-scenario pairs with DIFFERENT labels
    diff_mask = cs_lbls[nn_idx] != cs_lbls
    diff_indices = np.where(diff_mask)[0]
    top_diff = diff_indices[np.argsort(-nn_sims[diff_indices])]
    print("\nTop 10 closest cross-scenario pairs with DIFFERENT labels:")
    seen_pairs = set()
    printed = 0
    for i in top_diff:
        j = nn_idx[i]
        pair_key = tuple(sorted([cs_scens[i], cs_scens[j]]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        printed += 1
        print(f"  [{printed:02d}] sim={nn_sims[i]:.4f} | L{cs_lbls[i]}({cs_scens[i]}) <-> L{cs_lbls[j]}({cs_scens[j]})")
        print(f"       A: {repr(cs_df.loc[i, 'text'][:110])}")
        print(f"       B: {repr(cs_df.loc[j, 'text'][:110])}")
        if printed >= 10:
            break

    # ==================== 3. FINAL PHASE 3A MODEL & PREDICT test.csv ====================
    print("\n" + "=" * 70)
    print("3. TRAIN ON sample.csv (5x) + ALL synthetic_v2 -> submission_phase3a.csv")
    print("=" * 70)
    v2_proc = v2_df["text"].astype(str).apply(preprocess_text).tolist()
    v2_lbl = v2_df["label"].astype(int).values
    v2_tech = encode_techniques(v2_df["techniques"])
    v2_act = v2_df["action"].astype(int).values

    full_texts = real_clean + v2_proc
    full_lbl = np.concatenate([real_lbl, v2_lbl])
    full_tech = np.vstack([real_tech, v2_tech])
    full_act = np.concatenate([real_act, v2_act])
    full_weights = np.concatenate([np.full(len(real_lbl), 5.0), np.ones(len(v2_lbl))])

    char_v, word_v, X_full = fit_vectorizers(full_texts)
    clf_a = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_full, full_lbl, sample_weight=full_weights)
    clf_b = OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)).fit(X_full, full_tech)
    clf_c = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_full, full_act, sample_weight=full_weights)

    out_path = "submission_phase3a.csv"
    first_chunk = True
    label_counts = {k: 0 for k in range(6)}
    total_rows = 0

    for chunk in pd.read_csv("test.csv", chunksize=50000):
        clean_batch = chunk["text"].fillna("").astype(str).apply(preprocess_text).tolist()
        X_test = transform_vectorizers(char_v, word_v, clean_batch)
        probs_a = clf_a.predict_proba(X_test)
        preds_a = np.argmax(probs_a, axis=1)
        confs = np.max(probs_a, axis=1)
        tech_probs = clf_b.predict_proba(X_test)
        preds_c = clf_c.predict(X_test)

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
    print("Phase 3A Predicted Label Distribution on test.csv:")
    for k, v in label_counts.items():
        print(f"  Label {k}: {v:7,d} ({v/total_rows*100:5.2f}%)")

    print("\nValidating submission_phase3a.csv:")
    validate_submission(out_path)


if __name__ == "__main__":
    run_phase3a_check()
