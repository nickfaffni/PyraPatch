"""
download_datasets.py
Downloads all benchmark CSV datasets used by PyraPatch into PatchTST_supervised/dataset/.
Datasets from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
"""

import os
import subprocess
import sys


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


# Google Drive direct-download IDs for each dataset
DATASETS = {
    "ETTh1.csv":            "1l51LsnfakBif9FPrEpZLaAabDTfxuLOH",
    "ETTh2.csv":            "1F2OlAseOKnEk7jCrPmFuoMIcC8BjxmBJ",
    "ETTm1.csv":            "1ODsn2oxwFj5vAqXHnvub5y4V11yUlMaM",
    "ETTm2.csv":            "1nt_vz4RkClBanCDpSmAF_s0bFGiypHge",
    "weather.csv":          "1ZOS8TEZh3mT_cOdJCOV9CrBPRMlJsJoV",
    "exchange_rate.csv":    "1mLTtBaqMnJaKHOWzfXSuL_oJeCqdqdH1",
    "national_illness.csv": "1VJ8VPJgLGPFGdYVGJkLbHoVYl7pHGvoU",
    "electricity.csv":      "1uXMsZSNfPqfSAyMQEFmjliRPIoOcCPXP",
    "traffic.csv":          "1YZyvYLDp3XpmFcJMZerxa2r1XpAKnc2i",
}

SAVE_DIR = os.path.join(os.path.dirname(__file__), "PatchTST_supervised", "dataset")


def download_file(name, file_id, dest_dir):
    import gdown

    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest):
        print(f"  [skip] {name} already exists.")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading {name} ...")
    try:
        gdown.download(url, dest, quiet=False, fuzzy=True)
        print(f"  Done: {dest}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Please download manually from:")
        print(f"    https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy")


def main():
    ensure_gdown()
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving datasets to: {SAVE_DIR}\n")
    for name, fid in DATASETS.items():
        download_file(name, fid, SAVE_DIR)
    print("\nAll done! Place any missing files manually in PatchTST_supervised/dataset/")


if __name__ == "__main__":
    main()
