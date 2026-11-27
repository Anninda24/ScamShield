import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from validate_submission import validate_submission

ZIP_NAME = "NeuraHack_Preli_Shield.zip"
REQUIRED_FILES = ["submission.csv", "notebook.ipynb", "README.md", ".env"]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"C:/Users/", re.IGNORECASE),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def build_package(submission_path: str = "submission_v3.csv"):
    print(f"Validating {submission_path} against test.csv and sample_submission.csv...")
    if not validate_submission(submission_path):
        print("ABORT: Submission validation failed.")
        sys.exit(1)

    for req in ["notebook.ipynb", "README.md", ".env"]:
        if not os.path.isfile(req):
            print(f"ABORT: Missing required file: {req}")
            sys.exit(1)

    # Parse notebook.ipynb as JSON and check for API keys / local PC paths
    with open("notebook.ipynb", "r", encoding="utf-8") as f:
        nb_raw = f.read()
    nb_json = json.loads(nb_raw)
    n_cells = len(nb_json.get("cells", []))
    print(f"notebook.ipynb JSON parsed successfully: {n_cells} cells.")
    for pat in SECRET_PATTERNS:
        if pat.search(nb_raw):
            print(f"ABORT: Suspicious key or absolute path pattern found in notebook.ipynb: {pat.pattern}")
            sys.exit(1)
    print("Security & path check on notebook.ipynb: CLEAN (no API keys or PC paths found).")

    src_hash = sha256_file(submission_path)

    with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_path, arcname="submission.csv")
        zf.write("notebook.ipynb", arcname="notebook.ipynb")
        zf.write("README.md", arcname="README.md")
        zf.write(".env", arcname=".env")

    # Reopen and verify
    with zipfile.ZipFile(ZIP_NAME, "r") as zf:
        names = zf.namelist()
        assert names == REQUIRED_FILES, f"ZIP entries mismatch: {names} vs {REQUIRED_FILES}"
        zip_sub_hash = sha256_bytes(zf.read("submission.csv"))
        assert zip_sub_hash == src_hash, f"SHA-256 mismatch! src={src_hash}, zip={zip_sub_hash}"

    print(f"\nCreated {ZIP_NAME} successfully.")
    print("ZIP contents:", names)
    print(f"SHA-256 of submission.csv in ZIP == {submission_path}: {zip_sub_hash}")


if __name__ == "__main__":
    sub_arg = sys.argv[1] if len(sys.argv) > 1 else "submission_v3.csv"
    build_package(sub_arg)
