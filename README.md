
---

# PyraPatch: Hierarchical Multi-Scale Patching for Long-Term Time Series Forecasting

A research extension and implementation based on **"PatchTST"** (Nie et al., 2023) and **"Pyramid Vision Transformer"** (Wang et al., 2021).

---

## 📜 Academic Attribution & Credits

This project is an implementation of **PyraPatch**, a hierarchical extension of the PatchTST framework. The core conceptual logic and foundational architectures belong to the original authors of the following papers:

1. **PatchTST (Backbone)** *"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"* Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam | **ICLR 2023** | [Paper Link](https://arxiv.org/abs/2211.14730)
2. **Pyramid Vision Transformer (Hierarchical Inspiration)** *"Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions"* Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao | **ICCV 2021** | [Paper Link](https://arxiv.org/abs/2102.12122)

---

## 🚀 Overview

PyraPatch is designed to enhance **Long-term Time Series Forecasting (LTSF)** by introducing a hierarchical pyramid structure into the patching mechanism.

While the original PatchTST model relies on a fixed patch size—creating a static trade-off between local fluctuations and global trends—PyraPatch implements a multi-stage encoder with **1D Patch Merging** layers. This allows the model to learn multi-scale temporal representations, progressively increasing the receptive field while extracting deeper semantic features.

---

## 🧠 Architecture Logic

Our implementation adapts the "funnel" design from Computer Vision (PVT) for 1D time-series data:

* **High-Resolution Stage:** Initial processing of small patches () to capture high-frequency noise and local anomalies.
* **1D Patch Merging:** A custom layer that concatenates adjacent temporal tokens () and projects them into a higher-dimensional space ().
* **Global Trend Stage:** Deeper Transformer layers process aggregated tokens to model long-term seasonality and macroeconomic trends.

---

## 📊 Performance Benchmark

Benchmarks were conducted against the official PatchTST baseline. The hierarchical approach demonstrates significant gains in volatile, non-stationary environments.

| Dataset | Metric | Baseline (PatchTST) | PyraPatch (Ours) | Improvement |
| --- | --- | --- | --- | --- |
| **Exchange** | MSE | 0.0906 | **0.0829** | **+8.50% 🏆** |
| **ETTm1** | MSE | 0.3096 | **0.2988** | +3.50% |
| **ETTh1** | MSE | 0.3895 | **0.3842** | +1.38% |
| **Weather** | MSE | 0.1507 | **0.1504** | +0.20% |

> **Note:** Large-scale datasets (Traffic/Electricity) were excluded from this specific sweep due to local VRAM constraints.

---

## 🛠️ Installation & Usage

### Replicating the Exchange Experiment

```bash
python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id PyraPatch_Exchange \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --num_stages 3 \
  --patch_len 8 \
  --stride 4

```

---

## 👥 Implementation Team

* **Developer:** Nick Gaffni
* **Project Context:** Deep Learning Final Project (Academic Year 2025).

---

## 📖 BibTeX Citations

If you use this logic, please cite the original foundational works:

```bibtex
@inproceedings{nie2023patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{wang2021pvt,
  title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang eye and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}

```

---
