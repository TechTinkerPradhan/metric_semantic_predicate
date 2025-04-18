# Metric Semantic Predicate (MSP)

## Overview

`Metric Semantic Predicate (MSP)` is a modular system for training, evaluating, and updating Bayesian Neural Networks (BNNs) for spatial reasoning over 3D scene graphs. It decomposes spatial queries into metric, semantic, and predicate components and models each using Bayesian learning principles.

MSP supports query processing, model refinement with human feedback, and experiment tracking using TensorBoard and MLflow.

---

## Features

- **Modular BNN Models** for metric, semantic, predicate, and combined spatial query interpretation.
- **Bayesian Updating** using human-annotated datasets.
- **Scene Graph-Aware Query Processing** with custom PDF generation.
- **KDE + Wasserstein Evaluation** of model predictions.
- **MLflow + TensorBoard Logging** for traceable experimentation.

---

## Installation

```bash
git clone <your-repo-url>
cd final_msp
pip install -r requirements.txt
```

---

## Project Structure

```
metric_semantic_predicate/
├── dataset/
│   └── feature_utils.py             # Data prep, encoding, split
├── models/
│   ├── bayesian_combined_bnn.py     # Combined P_combined predictor
│   ├── bnn_metric_model.py          # Metric model (d0, sigma_m)
│   ├── bnn_semantic_model.py        # Semantic model (mu_x, mu_y, sigma_s)
│   └── bnn_predicate_model.py       # Predicate model (theta0, kappa)
├── scripts/
│   ├── train_model.py                          # Train combined model
│   ├── update_model_with_human.py             # Update combined model
│   ├── train_bnn_metric_model.py              # Train metric model
│   ├── update_bnn_metric_model_with_human.py
│   ├── train_bnn_semantic_model.py            # Train semantic model
│   ├── update_bnn_semantic_model_with_human.py
│   ├── train_bnn_predicate_model.py           # Train predicate model
│   └── update_bnn_predicate_model_with_human.py
├── training/
│   ├── evaluation_metrics.py        # KDE + Wasserstein distance
│   └── model_io.py                  # Save/load Pyro parameter store
├── utils/
│   ├── process_query.py
│   ├── pdf_generator.py
│   └── scene_graph_loader.py
├── configs/
│   ├── combined_pdf_bnn.yaml
│   ├── metric_bnn.yaml
│   ├── semantic_bnn.yaml
│   └── predicate_bnn.yaml
└── data/
    ├── 3DSceneGraph_Beechwood_dataset.csv
    ├── 3DSceneGraph_Beechwood_dataset_human.csv
    └── models/
```

---

## Usage

### Train Any Model

```bash
python3 -m metric_semantic_predicate.scripts.train_model \
  --config metric_semantic_predicate/configs/combined_pdf_bnn.yaml \
  --task combined
```

### Update Model with Human Data

```bash
python3 -m metric_semantic_predicate.scripts.update_bnn_predicate_model_with_human \
  --config metric_semantic_predicate/configs/predicate_bnn.yaml \
  --task predicate
```

---

## Logging and Evaluation

### View Training in TensorBoard

```bash
tensorboard --logdir runs/
```

Navigate to http://localhost:6006

### KDE & Wasserstein Distance (Python)

```python
from metric_semantic_predicate.training.evaluation_metrics import evaluate_with_distributions

# Compare predictions before and after update
evaluate_with_distributions(Y_true, mean_before, mean_after)
```

---

## License

MIT License © 2024 Swagat Padhan & Contributors
