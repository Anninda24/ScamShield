import os
import pandas as pd
from collections import Counter


def get_data_dir():
    return "data" if os.path.isdir("data") else "."


def inspect():
    data_dir = get_data_dir()
    print(f"Using data directory: {os.path.abspath(data_dir)}")

    sample = pd.read_csv(os.path.join(data_dir, "sample.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    for name, df in [("sample.csv", sample), ("test.csv", test), ("sample_submission.csv", sub)]:
        print(f"\n=== {name} ===")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Dtypes:\n", df.dtypes.to_dict())
        print("Nulls:\n", df.isna().sum().to_dict())

    print("\n=== sample.csv Label & Action Distributions ===")
    print("label counts:\n", sample["label"].value_counts().sort_index().to_dict())
    print("action counts:\n", sample["action"].value_counts().sort_index().to_dict())

    tech_counts = Counter()
    unsorted_techs = []
    separators_seen = set()
    for idx, val in enumerate(sample["techniques"].astype(str)):
        if val != "NONE":
            parts = val.split("|")
            if "|" in val:
                separators_seen.add("|")
            if parts != sorted(parts):
                unsorted_techs.append((sample.loc[idx, "id"], val))
            for p in parts:
                tech_counts[p] += 1
        else:
            tech_counts["NONE"] += 1

    print("techniques counts:", dict(sorted(tech_counts.items())))
    print("techniques separator(s):", separators_seen)
    print(f"unsorted techniques in sample.csv ({len(unsorted_techs)}):", unsorted_techs[:10])

    print("\n=== Label 0 and Label 5 Invariant Check in sample.csv ===")
    for lbl in [0, 5]:
        df_lbl = sample[sample["label"] == lbl]
        always_none = (df_lbl["techniques"] == "NONE").all()
        always_act0 = (df_lbl["action"] == 0).all()
        print(f"Label {lbl} (n={len(df_lbl)}): always techniques=='NONE' -> {always_none}, always action==0 -> {always_act0}")

    print("\n=== Conversation Structure & Text Lengths ===")
    for name, df in [("sample.csv", sample), ("test.csv", test)]:
        texts = df["text"].fillna("").astype(str)
        has_sep = texts.str.contains("|||", regex=False).sum()
        avg_len = texts.str.len().mean()
        print(f"{name}: rows containing '|||' = {has_sep:,} ({has_sep/len(df)*100:.2f}%), avg text length = {avg_len:.2f} chars")

    print("\n=== 5 Example Rows Per Class (sample.csv) ===")
    for lbl in sorted(sample["label"].unique()):
        print(f"\n--- Class {lbl} ---")
        subset = sample[sample["label"] == lbl].head(5)
        for _, r in subset.iterrows():
            print(f"  id={r['id']} | label={r['label']} | tech={r['techniques']} | action={r['action']} | text={repr(r['text'][:120])}")


if __name__ == "__main__":
    inspect()
