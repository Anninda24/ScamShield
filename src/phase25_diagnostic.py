import os
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

from preprocess import preprocess_text
from noise import apply_noise
from phase2_pipeline import (
    encode_techniques,
    fit_and_predict,
    compute_metrics,
    SEED,
)

# Map template family_id -> semantic scenario (class + scam story)
FAMILY_TO_SCENARIO = {
    # Class 0
    "c0_otp_warn": "0_otp_genuine_notice",
    "c0_tx_confirm": "0_tx_confirmation",
    "c0_parcel_legit": "0_delivery_genuine",
    "c0_mt_delivery": "0_delivery_genuine",
    "c0_mt_delivery_mix": "0_delivery_genuine",
    "c0_sec_alert": "0_security_warning",
    "c0_mt_support": "0_security_warning",
    "c0_bill_remind": "0_bill_reminder",
    "c0_cashback_real": "0_cashback_genuine",
    "c0_personal": "0_personal_chat",
    "c0_mt_chat": "0_personal_chat",
    # Class 1
    "c1_kyc_lock": "1_account_lock_kyc_pin",
    "c1_mt_bank": "1_account_lock_kyc_pin",
    "c1_mt_agent_bn": "1_account_lock_kyc_pin",
    "c1_mistake_tx": "1_mistaken_transfer",
    "c1_mt_agent": "1_mistaken_transfer",
    "c1_card_refund": "1_card_refund_cvv",
    # Class 2
    "c2_lottery_link": "2_lottery_cashback_link",
    "c2_mt_lottery_bn": "2_lottery_cashback_link",
    "c2_prize_fee": "2_prize_advance_fee",
    "c2_mt_prize": "2_prize_advance_fee",
    "c2_prize_app": "2_prize_spin_apk",
    "c2_prize_details": "2_prize_details_reply",
    # Class 3
    "c3_task_earn": "3_task_earning",
    "c3_mt_task": "3_task_earning",
    "c3_invest_fee": "3_investment_job_fee",
    "c3_mt_jobfee": "3_investment_job_fee",
    "c3_mt_long": "3_investment_job_fee",
    "c3_crypto": "3_crypto_vip_signals",
    "c3_abroad_job": "3_abroad_visa_fee",
    # Class 4
    "c4_parcel_fee": "4_parcel_storage_fee",
    "c4_customs": "4_parcel_customs_fee",
    "c4_mt_customs": "4_parcel_customs_fee",
    "c4_mt_customs_bn": "4_parcel_customs_fee",
    "c4_addr_link": "4_parcel_address_link",
    "c4_mt_addr_mix": "4_parcel_address_link",
    "c4_addr_call": "4_parcel_address_call",
    # Class 5
    "c5_promo_offer": "5_promo_sales_offer",
    "c5_mt_promo": "5_promo_sales_offer",
    "c5_mt_promo_bn": "5_promo_sales_offer",
    "c5_promo_app": "5_promo_app_install",
    "c5_promo_biz": "5_promo_real_estate",
    "c5_promo_course": "5_promo_course_ad",
}


def run_diagnostic():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    real_df = pd.read_csv("sample.csv")
    synth_df = pd.read_csv("data/synthetic.csv", keep_default_na=False)

    real_proc = real_df["text"].fillna("").astype(str).apply(preprocess_text).tolist()
    synth_proc = synth_df["text"].fillna("").astype(str).apply(preprocess_text).tolist()

    # ==================== 1. LEAKAGE CHECK ====================
    print("=" * 70)
    print("1. LEAKAGE CHECK (CHAR N-GRAM TF-IDF COSINE: 86 REAL vs 6,000 SYNTH)")
    print("=" * 70)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    X_synth = char_vec.fit_transform(synth_proc)
    X_real = char_vec.transform(real_proc)

    sim_matrix = cosine_similarity(X_real, X_synth)  # (86, 6000)
    max_sims = sim_matrix.max(axis=1)
    best_synth_idx = sim_matrix.argmax(axis=1)

    print(f"Max similarity stats across 86 real rows:")
    print(f"  Mean   : {max_sims.mean():.4f}")
    print(f"  Median : {np.median(max_sims):.4f}")
    print(f"  >= 0.6 : {(max_sims >= 0.6).mean() * 100:.2f}% ({(max_sims >= 0.6).sum()}/86)")
    print(f"  >= 0.8 : {(max_sims >= 0.8).mean() * 100:.2f}% ({(max_sims >= 0.8).sum()}/86)")

    top10_order = np.argsort(-max_sims)[:10]
    print("\nTop 10 closest real/synthetic pairs:")
    for rank, r_idx in enumerate(top10_order, 1):
        s_idx = best_synth_idx[r_idx]
        sim = max_sims[r_idx]
        print(f"  [{rank:02d}] sim={sim:.4f} | real(id={real_df.loc[r_idx, 'id']}, L={real_df.loc[r_idx, 'label']}) <-> synth(id={synth_df.loc[s_idx, 'id']}, fam={synth_df.loc[s_idx, 'family_id']}, L={synth_df.loc[s_idx, 'label']})")
        print(f"       REAL : {repr(real_df.loc[r_idx, 'text'][:110])}")
        print(f"       SYNTH: {repr(synth_df.loc[s_idx, 'text'][:110])}")

    # ==================== 2. VALIDATION (b) WITHOUT LEAKED FAMILIES ====================
    print("\n" + "=" * 70)
    print("2. VALIDATION (b) DROPPING FAMILIES WITH SIMILARITY >= 0.6 TO ANY REAL ROW")
    print("=" * 70)
    # Find every synthetic row with sim >= 0.6 to any real row
    leaked_synth_mask = (sim_matrix >= 0.6).any(axis=0)
    leaked_families = set(synth_df.loc[leaked_synth_mask, "family_id"].unique())
    keep_mask = ~synth_df["family_id"].isin(leaked_families)

    print(f"Leaked family_ids ({len(leaked_families)}): {sorted(leaked_families)}")
    print(f"Remaining synthetic rows: {keep_mask.sum()} / {len(synth_df)}")

    real_lbl = real_df["label"].astype(int).values
    real_tech = encode_techniques(real_df["techniques"])
    real_act = real_df["action"].astype(int).values

    synth_kept = synth_df[keep_mask].reset_index(drop=True)
    sk_proc = synth_kept["text"].apply(preprocess_text).tolist()
    sk_lbl = synth_kept["label"].astype(int).values
    sk_tech = encode_techniques(synth_kept["techniques"])
    sk_act = synth_kept["action"].astype(int).values

    pl_b, pt_b, pa_b, _ = fit_and_predict(sk_proc, sk_lbl, sk_tech, sk_act, real_proc)
    fa_b, fb_b, fc_b = compute_metrics(real_lbl, real_tech, real_act, pl_b, pt_b, pa_b)
    print(f"Validation (b) after dropping leaked families -> CLEAN Real 86: A={fa_b:.4f}, B={fb_b:.4f}, C={fc_b:.4f}")

    # ==================== 3 & 5. SCENARIO GROUPING & COUNT ====================
    print("\n" + "=" * 70)
    print("3 & 5. SCENARIO-LEVEL GROUPING & STRATIFIED GROUP K-FOLD")
    print("=" * 70)
    synth_df["scenario"] = synth_df["family_id"].map(FAMILY_TO_SCENARIO)
    assert synth_df["scenario"].isna().sum() == 0, "Unmapped family_id found!"

    print("Distinct scenarios per class in current generator:")
    for lbl in sorted(synth_df["label"].unique()):
        scens = sorted(synth_df[synth_df["label"] == lbl]["scenario"].unique())
        print(f"  Class {lbl}: {len(scens)} scenarios -> {scens}")

    synth_lbl = synth_df["label"].astype(int).values
    synth_tech = encode_techniques(synth_df["techniques"])
    synth_act = synth_df["action"].astype(int).values
    synth_scens = synth_df["scenario"].values
    synth_noisy_proc = [preprocess_text(apply_noise(t, rng, level=0.35)) for t in synth_df["text"].tolist()]

    sgkf = StratifiedGroupKFold(n_splits=5)
    d_cln_lbl, d_cln_tech, d_cln_act = np.zeros(len(synth_df), int), [""] * len(synth_df), np.zeros(len(synth_df), int)
    d_nsy_lbl, d_nsy_tech, d_nsy_act = np.zeros(len(synth_df), int), [""] * len(synth_df), np.zeros(len(synth_df), int)

    for tr_i, va_i in sgkf.split(synth_proc, synth_lbl, synth_scens):
        pl, pt, pa, _ = fit_and_predict(
            [synth_proc[i] for i in tr_i], synth_lbl[tr_i], synth_tech[tr_i], synth_act[tr_i],
            [synth_proc[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            d_cln_lbl[idx], d_cln_tech[idx], d_cln_act[idx] = pl[k], pt[k], pa[k]

        pl_n, pt_n, pa_n, _ = fit_and_predict(
            [synth_proc[i] for i in tr_i], synth_lbl[tr_i], synth_tech[tr_i], synth_act[tr_i],
            [synth_noisy_proc[i] for i in va_i]
        )
        for k, idx in enumerate(va_i):
            d_nsy_lbl[idx], d_nsy_tech[idx], d_nsy_act[idx] = pl_n[k], pt_n[k], pa_n[k]

    fa_s_c, fb_s_c, fc_s_c = compute_metrics(synth_lbl, synth_tech, synth_act, d_cln_lbl, d_cln_tech, d_cln_act)
    fa_s_n, fb_s_n, fc_s_n = compute_metrics(synth_lbl, synth_tech, synth_act, d_nsy_lbl, d_nsy_tech, d_nsy_act)
    print(f"\nScenario-Held-Out StratifiedGroupKFold:")
    print(f"  CLEAN: A={fa_s_c:.4f}, B={fb_s_c:.4f}, C={fc_s_c:.4f}")
    print(f"  NOISY: A={fa_s_n:.4f}, B={fb_s_n:.4f}, C={fc_s_n:.4f}")

    # ==================== 4. WORST SCENARIOS & 20 MOST COMMON ERRORS ====================
    print("\n" + "=" * 70)
    print("4. WORST-SCORING SCENARIOS (HELD-OUT) & TOP 20 ERRORS")
    print("=" * 70)
    scen_scores = []
    for scen in sorted(synth_df["scenario"].unique()):
        m = synth_df["scenario"] == scen
        acc = (d_cln_lbl[m] == synth_lbl[m]).mean()
        acc_n = (d_nsy_lbl[m] == synth_lbl[m]).mean()
        scen_scores.append((acc, acc_n, scen, int(m.sum()), int(synth_lbl[m][0])))

    scen_scores.sort(key=lambda x: x[0])
    print("Scenarios ranked worst to best (held-out Task A accuracy):")
    for acc, acc_n, scen, cnt, true_c in scen_scores[:15]:
        print(f"  {scen:26s} (Class {true_c}, n={cnt:3d}): CLEAN acc={acc:.4f} | NOISY acc={acc_n:.4f}")

    print("\n20 representative/common errors (CLEAN scenario-held-out):")
    err_indices = np.where(d_cln_lbl != synth_lbl)[0]
    # Count (true, pred) pairs and print top 20 diverse/frequent errors
    from collections import Counter
    pair_counts = Counter((int(synth_lbl[i]), int(d_cln_lbl[i])) for i in err_indices)
    print("Top (true -> pred) confusion pairs:", pair_counts.most_common(10))
    shown = 0
    for idx in err_indices:
        if shown >= 20:
            break
        print(f"  [{shown+1:02d}] true={synth_lbl[idx]} -> pred={d_cln_lbl[idx]} | scen={synth_scens[idx]} | text={repr(synth_df.loc[idx, 'text'][:100])}")
        shown += 1


if __name__ == "__main__":
    run_diagnostic()
