import os
import sys
import pandas as pd

REQUIRED_COLS = ["id", "label", "techniques", "action", "confidence"]
VALID_CODES = {"T1", "T2", "T3", "T4", "T5", "T6"}


def get_data_dir():
    return "data" if os.path.isfile(os.path.join("data", "sample.csv")) else "."


def check_techniques(val: str) -> bool:
    if not isinstance(val, str):
        return False
    if val != val.strip() or not val or " " in val:
        return False
    if val == "NONE":
        return True
    parts = val.split("|")
    if "NONE" in parts:
        return False
    if any(p not in VALID_CODES for p in parts):
        return False
    if len(parts) != len(set(parts)):
        return False
    if parts != sorted(parts):
        return False
    return True


def validate_submission(sub_path: str = "submission.csv") -> bool:
    data_dir = get_data_dir()
    ref_path = os.path.join(data_dir, "sample_submission.csv")
    test_path = os.path.join(data_dir, "test.csv")

    failures = []

    if not os.path.isfile(sub_path):
        print(f"FAIL: File not found: {sub_path}")
        return False

    sub = pd.read_csv(sub_path, keep_default_na=False)
    ref = pd.read_csv(ref_path, usecols=["id"])
    test = pd.read_csv(test_path, usecols=["id"])

    # 1. Header matches exactly
    if list(sub.columns) != REQUIRED_COLS:
        failures.append(f"Header mismatch: got {list(sub.columns)}, expected {REQUIRED_COLS}")

    # 2. Row count & ID set / order / duplicates
    if len(sub) != len(test):
        failures.append(f"Row count mismatch: got {len(sub)}, expected {len(test)}")
    if len(sub) != len(ref):
        failures.append(f"Row count mismatch vs sample_submission: got {len(sub)}, expected {len(ref)}")

    try:
        sub_ids = pd.to_numeric(sub["id"], errors="raise")
        if sub_ids.duplicated().any():
            failures.append(f"Duplicate IDs found: {sub_ids.duplicated().sum()}")
        if len(sub_ids) == len(test) and not (sub_ids.values == test["id"].values).all():
            failures.append("IDs do not match test.csv order or values")
    except Exception as e:
        failures.append(f"ID parsing error: {e}")

    # 3. No nulls or blank strings anywhere
    for col in REQUIRED_COLS:
        if col in sub.columns:
            blank_cnt = (sub[col].astype(str).str.strip() == "").sum()
            if blank_cnt > 0:
                failures.append(f"Column '{col}' has {blank_cnt} blank/null values")

    # 4. label and action are integers in 0-5
    for col in ["label", "action"]:
        if col in sub.columns:
            try:
                vals = pd.to_numeric(sub[col], errors="raise")
                if (vals % 1 != 0).any():
                    failures.append(f"Column '{col}' has non-integer values")
                if (~vals.isin([0, 1, 2, 3, 4, 5])).any():
                    failures.append(f"Column '{col}' has values outside 0-5")
            except Exception as e:
                failures.append(f"Column '{col}' parsing error: {e}")

    # 5. techniques validation
    if "techniques" in sub.columns:
        bad_tech = (~sub["techniques"].apply(check_techniques)).sum()
        if bad_tech > 0:
            failures.append(f"Column 'techniques' has {bad_tech} invalid entries")

    # 6. confidence numeric in [0, 1] and not NaN
    if "confidence" in sub.columns:
        try:
            confs = pd.to_numeric(sub["confidence"], errors="raise")
            if confs.isna().any() or ((confs < 0.0) | (confs > 1.0)).any():
                failures.append("Column 'confidence' has NaN or values outside [0, 1]")
        except Exception as e:
            failures.append(f"Column 'confidence' parsing error: {e}")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return False
    else:
        print("PASS")
        return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    ok = validate_submission(target)
    sys.exit(0 if ok else 1)
