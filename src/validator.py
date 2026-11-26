import pandas as pd

REQUIRED_COLS = ["id", "label", "techniques", "action", "confidence"]
VALID_LABELS = {0, 1, 2, 3, 4, 5}
VALID_ACTIONS = {0, 1, 2, 3, 4, 5}
VALID_TECHS = {"T1", "T2", "T3", "T4", "T5", "T6"}


def validate_techniques_cell(val: str) -> bool:
    if not isinstance(val, str):
        return False
    val = val.strip()
    if not val:
        return False
    if val == "NONE":
        return True
    parts = val.split("|")
    if len(parts) != len(set(parts)):
        return False
    return all(p in VALID_TECHS for p in parts)


def validate_submission(
    sub_path: str,
    sample_sub_path: str = "sample_submission.csv",
    test_path: str = "test.csv",
) -> dict:
    sub = pd.read_csv(sub_path, keep_default_na=False)
    ref_sub = pd.read_csv(sample_sub_path, usecols=["id"])
    test_ids = pd.read_csv(test_path, usecols=["id"])

    errors = []

    if list(sub.columns) != REQUIRED_COLS:
        errors.append(f"Column mismatch: got {list(sub.columns)}, expected {REQUIRED_COLS}")

    if len(sub) != len(ref_sub):
        errors.append(f"Row count mismatch: got {len(sub)}, expected {len(ref_sub)}")

    if len(sub) == len(test_ids) and not (sub["id"].astype(int).values == test_ids["id"].astype(int).values).all():
        errors.append("IDs do not match test.csv in exact order")

    if len(sub) == len(ref_sub) and not (sub["id"].astype(int).values == ref_sub["id"].astype(int).values).all():
        errors.append("IDs do not match sample_submission.csv in exact order")

    # Check empty strings / nulls
    for col in REQUIRED_COLS:
        empty_cnt = (sub[col].astype(str).str.strip() == "").sum()
        if empty_cnt > 0:
            errors.append(f"Column '{col}' has {empty_cnt} empty/null cells")

    # Check label
    try:
        labels = pd.to_numeric(sub["label"], errors="raise")
        if not labels.isin(VALID_LABELS).all():
            errors.append("Invalid label values outside {0..5}")
        if (labels % 1 != 0).any():
            errors.append("Non-integer label values found")
    except Exception as e:
        errors.append(f"Label parsing error: {e}")

    # Check action
    try:
        actions = pd.to_numeric(sub["action"], errors="raise")
        if not actions.isin(VALID_ACTIONS).all():
            errors.append("Invalid action values outside {0..5}")
        if (actions % 1 != 0).any():
            errors.append("Non-integer action values found")
    except Exception as e:
        errors.append(f"Action parsing error: {e}")

    # Check techniques
    invalid_tech = (~sub["techniques"].apply(validate_techniques_cell)).sum()
    if invalid_tech > 0:
        errors.append(f"Found {invalid_tech} invalid techniques cells")

    # Check confidence
    try:
        confs = pd.to_numeric(sub["confidence"], errors="raise")
        if ((confs < 0.0) | (confs > 1.0) | confs.isna()).any():
            errors.append("Confidence values outside [0.0, 1.0] or NaN")
    except Exception as e:
        errors.append(f"Confidence parsing error: {e}")

    # Check label == 0 invariant
    if not errors:
        lbl0_mask = sub["label"].astype(int) == 0
        bad_lbl0_tech = (sub.loc[lbl0_mask, "techniques"] != "NONE").sum()
        bad_lbl0_act = (sub.loc[lbl0_mask, "action"].astype(int) != 0).sum()
        if bad_lbl0_tech > 0 or bad_lbl0_act > 0:
            errors.append(f"Label 0 invariant violated: bad_tech={bad_lbl0_tech}, bad_act={bad_lbl0_act}")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "rows": len(sub),
        "errors": errors,
    }


if __name__ == "__main__":
    res = validate_submission("sample_submission.csv")
    print("sample_submission.csv validation:", res)
