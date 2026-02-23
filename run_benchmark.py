"""
run_benchmark.py  —  Cross-platform benchmark runner for PyraPatch vs Baseline.
Runs the standard ETT / Weather / Exchange-Rate suite across 3 seeds.
Run from PyraPatch/ root:  python run_benchmark.py
"""

import os
import sys

ROOT      = os.path.join(os.path.dirname(__file__), "PatchTST_supervised")
RUNNER    = os.path.join(ROOT, "run_longExp.py")
DATA_DIR  = os.path.join(ROOT, "dataset")

SEEDS = [2021, 2022, 2023]

# ── Dataset configurations ──────────────────────────────────────────────────
DATASETS = [
    # (data_flag, data_path, enc_in, freq)
    ("ETTh1",  "ETTh1.csv",           7,  "h"),
    ("ETTh2",  "ETTh2.csv",           7,  "h"),
    ("ETTm1",  "ETTm1.csv",           7,  "t"),
    ("ETTm2",  "ETTm2.csv",           7,  "t"),
    ("custom", "weather.csv",         21, "h"),
    ("custom", "exchange_rate.csv",    8, "h"),
]

PRED_LENS = [96, 192, 336, 720]


def run(model_id, dataset, data_path, enc_in, freq, pred_len, d_model, num_stages, seed):
    cmd = (
        f"{sys.executable} {RUNNER} "
        f"--random_seed {seed} "
        f"--is_training 1 "
        f"--model_id {model_id}_pl{pred_len}_s{seed} "
        f"--model PatchTST "
        f"--data {dataset} "
        f"--root_path {DATA_DIR}{os.sep} "
        f"--data_path {data_path} "
        f"--features M "
        f"--seq_len 336 "
        f"--label_len 48 "
        f"--pred_len {pred_len} "
        f"--e_layers 3 "
        f"--d_layers 1 "
        f"--factor 3 "
        f"--enc_in {enc_in} "
        f"--dec_in {enc_in} "
        f"--c_out {enc_in} "
        f"--d_model {d_model} "
        f"--d_ff 2048 "
        f"--num_stages {num_stages} "
        f"--batch_size 128 "
        f"--learning_rate 0.0001 "
        f"--train_epochs 100 "
        f"--patience 15 "
        f"--lradj type3 "
        f"--itr 1 "
        f"--des Exp"
    )
    print(f"\n>>> RUNNING: {model_id} | {data_path} | pred={pred_len} | seed={seed}")
    os.system(cmd)


def main():
    for dataset, data_path, enc_in, freq in DATASETS:
        for pred_len in PRED_LENS:
            for seed in SEEDS:
                # Baseline (flat, 1 stage)
                run(f"Baseline_{dataset}", dataset, data_path, enc_in, freq,
                    pred_len, 128, 1, seed)
                # PyraPatch (3 stages)
                run(f"PyraPatch_{dataset}", dataset, data_path, enc_in, freq,
                    pred_len, 128, 3, seed)

    print("\n\n🏆 Full benchmark finished! Check result.txt for metrics.")


if __name__ == "__main__":
    main()
