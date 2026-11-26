import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from preprocess import preprocess_text
from noise import apply_noise
from phase2_pipeline import (
    encode_techniques,
    fit_and_predict,
    compute_metrics,
    SEED,
)


def run_phase3a_eval():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    df = pd.read_csv("data/synthetic_v2.csv", keep_default_na=False)
    print(f"Loaded data/synthetic_v2.csv: {df.shape}")

    print("\n=== Scenarios per Class (synthetic_v1 + synthetic_v2) ===")
    synth_only = df[df["source"] != "real_paraphrase"].copy()
    for lbl in range(6):
        scens = sorted(synth_only[synth_only["label"] == lbl]["scenario_id"].unique())
        print(f"  Class {lbl}: {len(scens)} scenarios")

    # Scenario-held-out StratifiedGroupKFold grouped by scenario_id
    # Note: evaluate on synthetic scenarios (or full v2 where paraphrases are grouped by real_para_<id>)
    proc_clean = df["text"].astype(str).apply(preprocess_text).tolist()
    proc_noisy = [preprocess_text(apply_noise(t, rng, level=0.35)) for t in df["text"].astype(str)]

    y_lbl = df["label"].astype(int).values
    y_tech = encode_techniques(df["techniques"])
    y_act = df["action"].astype(int).values
    groups = df["scenario_id"].values

    sgkf = StratifiedGroupKFold(n_splits=5)
    p_cln_lbl, p_cln_tech, p_cln_act = np.zeros(len(df), int), [""] * len(df), np.zeros(len(df), int)
    p_nsy_lbl, p_nsy_tech, p_nsy_act = np.zeros(len(df), int), [""] * len(df), np.zeros(len(df), int)

    for fold, (tr_i, va_i) in enumerate(sgkf.split(proc_clean, y_lbl, groups)):
        pl, pt, pa, _ = fit_and_predict(
            [proc_clean[i] for i in tr_i], y_lbl[tr_i], y_tech[tr_i], y_act[tr_i],
            [proc_clean[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            p_cln_lbl[idx], p_cln_tech[idx], p_cln_act[idx] = pl[k], pt[k], pa[k]

        pl_n, pt_n, pa_n, _ = fit_and_predict(
            [proc_clean[i] for i in tr_i], y_lbl[tr_i], y_tech[tr_i], y_act[tr_i],
            [proc_noisy[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            p_nsy_lbl[idx], p_nsy_tech[idx], p_nsy_act[idx] = pl_n[k], pt_n[k], pa_n[k]

    fa_c, fb_c, fc_c = compute_metrics(y_lbl, y_tech, y_act, p_cln_lbl, p_cln_tech, p_cln_act)
    fa_n, fb_n, fc_n = compute_metrics(y_lbl, y_tech, y_act, p_nsy_lbl, p_nsy_tech, p_nsy_act)

    print("\n=== Phase 3A Scenario-Held-Out StratifiedGroupKFold (grouped by scenario_id) ===")
    print(f"All v2 (n={len(df)}):")
    print(f"  CLEAN: Task A={fa_c:.4f}, Task B={fb_c:.4f}, Task C={fc_c:.4f}")
    print(f"  NOISY: Task A={fa_n:.4f}, Task B={fb_n:.4f}, Task C={fc_n:.4f}")

    # Also report strictly on synthetic scenarios (excluding real_paraphrase from eval)
    m_s = (df["source"] != "real_paraphrase").values
    fa_sc, fb_sc, fc_sc = compute_metrics(
        y_lbl[m_s], y_tech[m_s], y_act[m_s],
        p_cln_lbl[m_s], [p_cln_tech[i] for i in np.where(m_s)[0]], p_cln_act[m_s]
    )
    fa_sn, fb_sn, fc_sn = compute_metrics(
        y_lbl[m_s], y_tech[m_s], y_act[m_s],
        p_nsy_lbl[m_s], [p_nsy_tech[i] for i in np.where(m_s)[0]], p_nsy_act[m_s]
    )
    print(f"Synthetic scenarios only (n={m_s.sum()}, excluding real_paraphrase):")
    print(f"  CLEAN: Task A={fa_sc:.4f}, Task B={fb_sc:.4f}, Task C={fc_sc:.4f}")
    print(f"  NOISY: Task A={fa_sn:.4f}, Task B={fb_sn:.4f}, Task C={fc_sn:.4f}")

    # Print 40 random new rows (at least 10 from new synthetic_v2 scenarios)
    print("\n=== 40 Random Rows from data/synthetic_v2.csv (including >= 10 new v2 scenarios) ===")
    v2_new = df[df["source"] == "synthetic_v2"].sample(20, random_state=SEED)
    para_s = df[df["source"] == "real_paraphrase"].sample(10, random_state=SEED)
    v1_s = df[df["source"] == "synthetic_v1"].sample(10, random_state=SEED)
    qc_df = pd.concat([v2_new, para_s, v1_s]).sample(frac=1.0, random_state=SEED)

    for i, (_, r) in enumerate(qc_df.iterrows(), 1):
        print(f"[{i:02d}] src={r['source']:15s} | scen={r['scenario_id']:26s} | L={r['label']} | T={r['techniques']:8s} | A={r['action']} | {r['language']:8s} | turns={r['n_turns']} | noisy={r['is_noisy']}")
        print(f"     {repr(str(r['text'])[:135])}")


if __name__ == "__main__":
    run_phase3a_eval()
