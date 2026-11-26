import hashlib
import os
import sys
import zipfile

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from validate_submission import validate_submission

ZIP_NAME = "NeuraHack_Preli_Shield.zip"
REQUIRED_FILES = ["submission.csv", "notebook.ipynb", "README.md", ".env"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def build_package(submission_path: str = "submission.csv"):
    print(f"Validating {submission_path}...")
    if not validate_submission(submission_path):
        print("ABORT: Submission validation failed.")
        sys.exit(1)

    for req in ["notebook.ipynb", "README.md", ".env"]:
        if not os.path.isfile(req):
            print(f"ABORT: Missing required file: {req}")
            sys.exit(1)

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
    print(f"SHA-256 verified (byte-identical): {zip_sub_hash}")


if __name__ == "__main__":
    sub_arg = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    build_package(sub_arg)
