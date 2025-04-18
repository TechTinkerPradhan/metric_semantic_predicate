import numpy as np
import torch
import pandas as pd
from scipy.stats import wasserstein_distance
import pyro

from metric_semantic_predicate.training.model_io import load_model
from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset

from metric_semantic_predicate.models.bnn_metric_model import (
    bnn_predict_metric, BayesianNN_metric, guide_bnn_metric
)
from metric_semantic_predicate.models.bnn_semantic_model import (
    bnn_predict_sem, BayesianNNSem, guide_bnn_sem
)
from metric_semantic_predicate.models.bnn_predicate_model import (
    bnn_predict_predicate, BayesianNNPred, guide_bnn_pred
)
from metric_semantic_predicate.models.bnn_combined_model import (
    bnn_predict, BayesianNNcombined
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

def combined_pdf(x0, y0, z0, params):
    mu = np.array([params['mu_x'], params['mu_y'], params['mu_z']])
    loc = np.array([x0, y0, z0])
    distance = np.linalg.norm(loc - mu)
    exponent = -((distance - params['d0']) ** 2) / (2 * (params['sigma_m'] + params['sigma_s']) ** 2 + 1e-6)
    return np.exp(exponent)

def compute_analytical_combined_pdf(X_input_metric, X_input_sem, X_input_pred, net_metric, net_sem, net_pred):
    mean_metric, _ = bnn_predict_metric(X_input_metric, net_metric)
    mean_sem, _ = bnn_predict_sem(X_input_sem, net_sem)
    mean_pred, _ = bnn_predict_predicate(X_input_pred, net_pred)

    d0, sigma_m = mean_metric[:, 0], mean_metric[:, 1]
    mu_x, mu_y, mu_z, sigma_s = mean_sem[:, 0], mean_sem[:, 1], mean_sem[:, 2], mean_sem[:, 3]
    theta0, phi0, kappa = mean_pred[:, 0], mean_pred[:, 1], mean_pred[:, 2]

    P_combined_values = []
    for i in range(len(X_input_metric)):
        params = {
            'mu_x': mu_x[i], 'mu_y': mu_y[i], 'mu_z': mu_z[i], 'sigma_s': sigma_s[i],
            'x0': X_input_metric[i, 0], 'y0': X_input_metric[i, 1], 'z0': X_input_metric[i, 2],
            'd0': d0[i], 'sigma_m': sigma_m[i],
            'theta0': theta0[i], 'phi0': phi0[i], 'kappa': kappa[i]
        }
        P_combined_values.append(
            combined_pdf(
                params['x0'].cpu().item(),
                params['y0'].cpu().item(),
                params['z0'].cpu().item(),
                {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in params.items()}
            )
        )
    return np.array(P_combined_values)

if __name__ == "__main__":
    # 🔹 Clear param store
    pyro.clear_param_store()

    # Load dataset and split
    df = pd.read_csv("metric_semantic_predicate/data/3DSceneGraph_Beechwood_dataset.csv")
    _, _, df_test = split_dataset(df)

    # Define column sets
    metric_cols = ["metric", "width", "height", "depth"]
    semantic_cols = ["encoded_semantic", "width", "height", "depth"]
    predicate_cols = ["encoded_predicate", "width", "height", "depth"]
    combined_cols = ["metric", "encoded_predicate", "encoded_semantic", "width", "height", "depth"]
    target_cols = ["P_combined"]

    # Prepare test sets
    X_metric, _, _, _, _ = prepare_dataset(df_test, metric_cols, ["d0", "sigma_m"])
    X_sem, _, _, _, _ = prepare_dataset(df_test, semantic_cols, ["mu_x", "mu_y", "mu_z", "sigma_s"])
    X_pred, _, _, _, _ = prepare_dataset(df_test, predicate_cols, ["theta0", "phi0", "kappa"])
    X_comb, Y_comb, _, _, _ = prepare_dataset(df_test, combined_cols, target_cols)

    # Convert to tensors
    X_metric_t = torch.tensor(X_metric, dtype=torch.float32).to(device)
    X_sem_t = torch.tensor(X_sem, dtype=torch.float32).to(device)
    X_pred_t = torch.tensor(X_pred, dtype=torch.float32).to(device)
    X_comb_t = torch.tensor(X_comb, dtype=torch.float32).to(device)
    Y_comb_np = Y_comb[:, 0]

    # Init model architectures
    net_metric = BayesianNN_metric(input_dim=4, hidden_dim=32, output_dim=2).to(device)
    net_sem = BayesianNNSem(input_dim=4, hidden_dim=32, output_dim=4).to(device)
    net_pred = BayesianNNPred(input_dim=4, hidden_dim=32, output_dim=3).to(device)
    net_combined = BayesianNNcombined(input_dim=6, hidden_dim=32, output_dim=1).to(device)

    # 🔹 Correct Model loading sequence (CRITICAL FIX)
    
    # Metric Model
    load_model("bnn_metric_after_update", save_dir="data/models")
    _ = guide_bnn_metric(torch.zeros(1, 4).to(device), torch.zeros(1, 2).to(device), net_metric)

    # Semantic Model
    load_model("sem_bnn_after_update", save_dir="data/models")
    _ = guide_bnn_sem(torch.zeros(1, 4).to(device), torch.zeros(1, 4).to(device), net_sem)

    # Predicate Model
    load_model("predicate_bnn_after_update", save_dir="data/models")
    _ = guide_bnn_pred(torch.zeros(1, 4).to(device), torch.zeros(1, 3).to(device), net_pred)


    # Combined Model (no guide needed if directly predicting)
    load_model("combined_bnn_after_update", save_dir="data/models")

    # 🔹 Predict and Compare
    print("🔮 Computing analytical PDF from submodels...")
    P_combined_analytical = compute_analytical_combined_pdf(
        X_metric_t, X_sem_t, X_pred_t, net_metric, net_sem, net_pred
    )

    print("🧠 Predicting using combined BNN...")
    mean_combined, _ = bnn_predict(X_comb_t, net_combined)
    mean_combined_np = mean_combined.squeeze()


    # 🔹 Evaluation
    print("\n📊 Evaluation Report")
    print(f"Wasserstein Distance (Analytical vs. Ground Truth): {wasserstein_distance(P_combined_analytical, Y_comb_np):.4f}")
    print(f"Wasserstein Distance (Combined BNN vs. Ground Truth): {wasserstein_distance(mean_combined_np, Y_comb_np):.4f}")
    print(f"Wasserstein Distance (Combined BNN vs. Analytical): {wasserstein_distance(mean_combined_np, P_combined_analytical):.4f}")
