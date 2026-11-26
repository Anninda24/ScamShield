import pandas as pd
import re


def run_eda():
    print("=" * 60)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS (REAL CSV VERIFICATION)")
    print("=" * 60)

    sample = pd.read_csv("sample.csv")
    test = pd.read_csv("test.csv")
    sub = pd.read_csv("sample_submission.csv")

    print("\n--- 1. File Shapes & Columns ---")
    print(f"sample.csv            : {sample.shape} | cols={sample.columns.tolist()}")
    print(f"test.csv              : {test.shape} | cols={test.columns.tolist()}")
    print(f"sample_submission.csv : {sub.shape} | cols={sub.columns.tolist()}")

    print("\n--- 2. Missing Values & ID Checks ---")
    print("sample.csv nulls:\n", sample.isna().sum().to_dict())
    print("test.csv nulls:\n", test.isna().sum().to_dict())
    print("sample_submission.csv nulls:\n", sub.isna().sum().to_dict())
    print(f"sample ID range: {sample['id'].min()} .. {sample['id'].max()} (unique={sample['id'].nunique()})")
    print(f"test ID range  : {test['id'].min()} .. {test['id'].max()} (unique={test['id'].nunique()})")
    print(f"test IDs == sample_submission IDs: {(test['id'].values == sub['id'].values).all()}")

    print("\n--- 3. sample.csv Label / Technique / Action Stats ---")
    print("Label counts:")
    for lbl, cnt in sample["label"].value_counts().sort_index().items():
        print(f"  Label {lbl}: {cnt:3d} ({cnt/len(sample)*100:5.1f}%)")

    print("\nAction counts:")
    for act, cnt in sample["action"].value_counts().sort_index().items():
        print(f"  Action {act}: {cnt:3d} ({cnt/len(sample)*100:5.1f}%)")

    print("\nTechniques counts (individual codes):")
    from collections import Counter
    t_counter = Counter()
    for t_str in sample["techniques"].fillna(""):
        for t in str(t_str).split("|"):
            t_counter[t] += 1
    for k, v in sorted(t_counter.items()):
        print(f"  {k:5s}: {v:3d}")

    print("\n--- 4. Checking Label 0 and Label 5 Invariants in sample.csv ---")
    for target_lbl in [0, 5]:
        sub_df = sample[sample["label"] == target_lbl]
        print(f"Label {target_lbl} (n={len(sub_df)}):")
        print("  unique techniques:", sub_df["techniques"].value_counts().to_dict())
        print("  unique actions   :", sub_df["action"].value_counts().to_dict())

    print("\n--- 5. Text Characteristics (sample.csv vs test.csv) ---")
    for name, df in [("sample.csv", sample), ("test.csv", test)]:
        texts = df["text"].fillna("").astype(str)
        lens = texts.str.len()
        multi_turn = texts.str.contains(r"\|\|\|", regex=True).sum()
        has_bn = texts.str.contains(r"[\u0980-\u09FF]", regex=True).sum()
        has_zwsp = texts.str.contains("\u200b", regex=False).sum()
        has_zwnj = texts.str.contains("\u200c", regex=False).sum()
        has_zwj = texts.str.contains("\u200d", regex=False).sum()
        has_bom = texts.str.contains("\ufeff", regex=False).sum()
        has_defang = texts.str.contains(r"\[\.\]", regex=True).sum()
        print(f"{name}:")
        print(f"  len: min={lens.min()}, median={lens.median():.1f}, mean={lens.mean():.1f}, max={lens.max()}")
        print(f"  unique texts: {texts.nunique():,} / {len(df):,}")
        print(f"  multi-turn (|||): {multi_turn:,} ({multi_turn/len(df)*100:.2f}%)")
        print(f"  Bangla script   : {has_bn:,} ({has_bn/len(df)*100:.2f}%)")
        print(f"  defanged [.]    : {has_defang:,} ({has_defang/len(df)*100:.2f}%)")
        print(f"  ZWSP (\\u200b)   : {has_zwsp:,} ({has_zwsp/len(df)*100:.2f}%)")
        print(f"  ZWNJ (\\u200c)   : {has_zwnj:,} ({has_zwnj/len(df)*100:.2f}%)")
        print(f"  ZWJ  (\\u200d)   : {has_zwj:,} ({has_zwj/len(df)*100:.2f}%)")
        print(f"  BOM  (\\ufeff)   : {has_bom:,} ({has_bom/len(df)*100:.2f}%)")


if __name__ == "__main__":
    run_eda()
