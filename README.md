# PyraPatch: Hierarchical Patch Merging for Long-Term Time Series Forecasting

PyraPatch extends PatchTST ([Nie et al., ICLR 2023](https://arxiv.org/abs/2211.14730)) with a
**multi-stage pyramid architecture** that progressively merges adjacent patch tokens, doubling the
receptive field at each stage while halving sequence length — analogous to Swin Transformer's
patch-merging in vision.

---

## Key Idea

| Model | Stages | Patch Flow |
|-------|--------|------------|
| **Baseline** | 1 | Flat Transformer, fixed patch size |
| **PyraPatch** | 3 | Stage 0 → PatchMerging → Stage 1 → PatchMerging → Stage 2 |

At each merging step, two adjacent patch tokens are concatenated and projected, producing a
**coarser but richer** representation.

---

## Setup

### Using uv (Recommended)

```bash
# 1. Install dependencies
uv sync

# 2. Download datasets
uv run download_datasets.py
```

### Using pip

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python download_datasets.py
```

If `download_datasets.py` fails (Google Drive limits), download manually from
[Autoformer GDrive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
and place CSV files in `PatchTST_supervised/dataset/`.

---

## Running Experiments

### ETT / Standard Benchmark

```bash
uv run run_benchmark.py
# or: python run_benchmark.py
```

Runs Baseline (1 stage) vs PyraPatch (3 stages) on ETTh1, ETTh2, ETTm1, ETTm2,
Weather, Exchange-Rate — 4 prediction horizons × 3 seeds.

### ECL Benchmark (High-Dimensional, 321 vars)

```bash
uv run run_ecl.py
# or: python run_ecl.py
```

> **GPU Required**: ≥12 GB VRAM. batch_size=8 is used to prevent OOM.

Runs: Baseline (d=128), BaselineLarge (d=256), PyraPatch (d=128, 3 stages) × 3 seeds.

### Single Experiment

```bash
# Baseline
uv run PatchTST_supervised/run_longExp.py \
  --is_training 1 --model_id Baseline_ETTh1 --model PatchTST \
  --data ETTh1 --root_path ./PatchTST_supervised/dataset/ --data_path ETTh1.csv \
  --features M --seq_len 336 --pred_len 96 --enc_in 7 \
  --e_layers 3 --d_model 128 --patch_len 16 --stride 8 \
  --num_stages 1 --itr 1 --patience 10

# PyraPatch
uv run PatchTST_supervised/run_longExp.py \
  --is_training 1 --model_id PyraPatch_ETTh1 --model PatchTST \
  --data ETTh1 --root_path ./PatchTST_supervised/dataset/ --data_path ETTh1.csv \
  --features M --seq_len 336 --pred_len 96 --enc_in 7 \
  --e_layers 3 --d_model 128 --patch_len 8 --stride 4 \
  --num_stages 3 --itr 1 --patience 10
```

Results are written to `result.txt`.

---

## Project Structure

```
PyraPatch/
├── PatchTST_supervised/
│   ├── models/         PatchTST.py (PyraPatch backbone)
│   ├── layers/         Transformer_EncDec, SelfAttention_Family, Embed, RevIN
│   ├── exp/            Exp_Main (PatchTST-only), Exp_Basic
│   ├── data_provider/  Dataset loaders (ETT, Custom)
│   ├── utils/          tools, metrics, masking, timefeatures
│   ├── dataset/        CSV files (download_datasets.py)
│   └── run_longExp.py  Main training entry point
├── download_datasets.py
├── run_benchmark.py
├── run_ecl.py
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@inproceedings{Yuqietal-2023-PatchTST,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and H. Nguyen, Nam and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```
