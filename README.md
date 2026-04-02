# GNN Fraud Detection

A graph-based fraud detection system built with PyTorch Geometric, trained on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) synthetic financial transaction dataset. This project implements a GraphSAGE model for edge-level fraud classification and compares it against traditional machine learning baselines.

---

## Project Overview

Financial fraud detection is naturally a graph problem, accounts are nodes, transactions are edges, and fraudulent behavior often forms detectable patterns in the network topology. This project explores whether graph neural networks can exploit that structure better than feature-only baselines.

---
## Pipeline Overview

![Project Summary](project_summary.svg)
---

## Graph Design

| Component | Meaning |
|---|---|
| **Node** | A financial account (`nameOrig` or `nameDest`) |
| **Edge** | A transaction between two accounts |
| **Edge label** | `1` = fraud, `0` = legitimate |
| **Node features** | 12 behavioral aggregates (see below) |

### Node Features (12 dimensions)

Each account node is described by aggregated statistics computed from its transaction history — no label information is used.

| Feature | Description |
|---|---|
| `tx_count` | Number of transactions sent |
| `avg_amount` | Average transaction amount sent |
| `max_amount` | Maximum transaction amount sent |
| `total_amount` | Total amount sent |
| `std_amount` | Standard deviation of amounts sent |
| `unique_dest` | Number of unique recipients |
| `time_span` | Time range of activity (max step - min step) |
| `dest_ratio` | Ratio of unique destinations to total transactions |
| `recv_count` | Number of transactions received |
| `recv_avg` | Average amount received |
| `recv_max` | Maximum amount received |
| `unique_senders` | Number of unique senders to this account |

All features are normalized to [0, 1].


## Training Details

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` with `pos_weight=10.0` |
| Optimizer | Adam, lr=0.005, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=25) |
| Epochs | 300 |
| Threshold | Swept over [0.10, 0.95] to maximize F1 |
| Split | 80/20 stratified train/test |

---

## Results

### GNN Training Progression

| Epoch | Loss | Precision | Recall | F1 |
|---|---|---|---|---|
| 0 | 1.263 | 0.091 | 1.000 | 0.167 |
| 60 | 1.005 | 0.539 | 0.415 | 0.469 |
| 160 | 0.979 | 0.568 | 0.405 | 0.473 |
| 260 | 0.966 | 0.552 | 0.417 | **0.475** |

### Final Comparison

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.371 | 0.506 | 0.428 |
| XGBoost | 0.286 | 0.575 | 0.382 |
| **GNN — GraphSAGE** | **0.553** | **0.417** | **0.475** |
| **Random Forest** | **0.625** | **0.740** | **0.677** |

### What This Means

The GNN has access to *more* information than the baselines (graph topology + features) but scores lower than Random Forest. This is not evidence that graph structure is useless, it reflects that:

1. **Node features computed from the full sample are already highly informative.** Random Forest exploits them directly without needing to propagate through the graph.
2. **Basic GNN architectures need richer graph structure to shine.** In production settings where new accounts have sparse history, neighborhood aggregation would provide a stronger advantage.
3. **GNN beats Logistic Regression and XGBoost**, showing it does extract meaningful signal beyond simple linear combinations.

 

## References

- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.* EMSS 2016.
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. (GraphSAGE)
- Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop 2019.
