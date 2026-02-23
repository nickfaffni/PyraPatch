"""
run_ecl.py  —  Cross-platform ECL (Electricity) benchmark for PyraPatch.
Runs 3 models × 3 seeds on the ECL dataset (321 variates).
Run from PyraPatch/ root:  python run_ecl.py

> [!IMPORTANT]
> batch_size=8 is required to avoid OOM errors with 321 channels.
> You need a GPU with at least 12 GB VRAM (24 GB recommended).
"""

import os
import sys

ROOT     = os.path.join(os.path.dirname(__file__), "PatchTST_supervised")
RUNNER   = os.path.join(ROOT, "run_longExp.py")
DATA_DIR = os.path.join(ROOT, "dataset")

SEEDS = [2021, 2022, 2023]


def run_experiment(model_id, d_model, num_stages, seed):
    cmd = (
        f"{sys.executable} {RUNNER} "
        f"--random_seed {seed} "
        f"--is_training 1 "
        f"--model_id {model_id}_s{seed} "
        f"--model PatchTST "
        f"--data custom "
        f"--root_path {DATA_DIR}{os.sep} "
        f"--data_path electricity.csv "
        f"--features M "
        f"--seq_len 336 "
        f"--label_len 48 "
        f"--pred_len 96 "
        f"--e_layers 3 "
        f"--d_layers 1 "
        f"--factor 3 "
        f"--enc_in 321 "
        f"--dec_in 321 "
        f"--c_out 321 "
        f"--d_model {d_model} "
        f"--d_ff 2048 "
        f"--batch_size 8 "
        f"--learning_rate 0.0001 "
        f"--num_stages {num_stages} "
        f"--train_epochs 100 "
        f"--patience 15 "
        f"--lradj type3 "
        f"--itr 1 "
        f"--des Exp"
    )
    print(f"\n>>> MISSION CONTROL: Starting {model_id} | Seed: {seed}")
    os.system(cmd)


def main():
    for seed in SEEDS:
        # 1. Standard Baseline (Flat Architecture, d_model=128)
        run_experiment("Baseline_ECL", 128, 1, seed)

        # 2. Large Baseline (Parameter Scaling Check, d_model=256)
        run_experiment("BaselineLarge_ECL", 256, 1, seed)

        # 3. PyraPatch (Proposed Hierarchical Model, d_model=128, 3 stages)
        run_experiment("PyraPatch_ECL", 128, 3, seed)

    print("\n\n🏆 All Electricity experiments are finished! Check result.txt for metrics.")


if __name__ == "__main__":
    main()
