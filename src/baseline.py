import os
import random
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from preprocess import preprocess_text
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
                best_j = int(np.argmax(tech_probs[i]))
                active = [TECH_CODES[best_j]]
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


def run_baseline():
    set_seed(SEED)
    t0 = time.time()
    data_dir = get_data_dir()

    sample = pd.read_csv(os.path.join(data_dir, "sample.csv"))
    texts = sample["text"].fillna("").astype(str).apply(preprocess_text).tolist()
    y_label = sample["label"].astype(int).values
    y_tech = encode_techniques(sample["techniques"])
    y_action = sample["action"].astype(int).values

    # Stratified 5-fold CV on sample.csv
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_label = np.zeros(len(sample), dtype=int)
    oof_tech_probs = np.zeros_like(y_tech, dtype=float)
    oof_action = np.zeros(len(sample), dtype=int)
    oof_conf = np.zeros(len(sample), dtype=float)

    for tr_idx, va_idx in skf.split(texts, y_label):
        char_v, word_v, X_tr = fit_vectorizers([texts[i] for i in tr_idx])
        X_va = transform_vectorizers(char_v, word_v, [texts[i] for i in va_idx])

        clf_a = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        clf_a.fit(X_tr, y_label[tr_idx])
        probs_a = clf_a.predict_proba(X_va)
        oof_label[va_idx] = np.argmax(probs_a, axis=1)
        oof_conf[va_idx] = np.max(probs_a, axis=1)

        clf_b = OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        )
        clf_b.fit(X_tr, y_tech[tr_idx])
        oof_tech_probs[va_idx] = clf_b.predict_proba(X_va)

        clf_c = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        clf_c.fit(X_tr, y_action[tr_idx])
        oof_action[va_idx] = clf_c.predict(X_va)

    oof_label, oof_tech_strs, oof_action, oof_conf = postprocess_predictions(
        oof_label, oof_tech_probs, oof_action, oof_conf
    )
    oof_tech_bin = encode_techniques(pd.Series(oof_tech_strs))

    f1_a = f1_score(y_label, oof_label, average="macro")
    f1_b = float(np.mean([f1_score(y_tech[:, j], oof_tech_bin[:, j], zero_division=0) for j in range(6)]))
    f1_c = f1_score(y_action, oof_action, average="macro")

    print("=== 5-Fold Stratified CV on sample.csv (optimistic due to near-duplicate templates) ===")
    print(f"Task A Macro F1: {f1_a:.4f}")
    print(f"Task B Macro F1: {f1_b:.4f}")
    print(f"Task C Macro F1: {f1_c:.4f}")

    # Train final model on all of sample.csv
    char_vec, word_vec, X_full = fit_vectorizers(texts)

    full_clf_a = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    full_clf_a.fit(X_full, y_label)

    full_clf_b = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    )
    full_clf_b.fit(X_full, y_tech)

    full_clf_c = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    full_clf_c.fit(X_full, y_action)

    # Predict test.csv in chunks and write submission.csv
    test_path = os.path.join(data_dir, "test.csv")
    out_path = "submission.csv"
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

        preds_a, tech_strs, preds_c, confs = postprocess_predictions(
            preds_a, tech_probs, preds_c, confs
        )

        for lbl in preds_a:
            label_counts[int(lbl)] += 1

        df_out = pd.DataFrame({
            "id": chunk["id"].values.astype(int),
            "label": preds_a.astype(int),
            "techniques": tech_strs,
            "action": preds_c.astype(int),
            "confidence": confs,
        })
        df_out.to_csv(out_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(chunk)

    print(f"\nSaved {out_path} ({total_rows:,} rows) in {time.time() - t0:.2f}s")
    print("Predicted Label Distribution on test.csv:")
    for k, v in label_counts.items():
        print(f"  Label {k}: {v:7,d} ({v/total_rows*100:5.2f}%)")

    print("\nRunning validator on submission.csv:")
    validate_submission(out_path)


if __name__ == "__main__":
    run_baseline()
