import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import torch

def evaluate_with_distributions(Y_true, mean_before, mean_after, label_prefix="BNN"):
    """
    Evaluate model predictions using Wasserstein distance and plot distribution comparisons.

    Parameters:
    - Y_true: Ground truth values (Tensor or np.ndarray) 
    - mean_before: Predicted means before update
    - mean_after: Predicted means after update
    - label_prefix: Prefix for labeling plots
    """
    Y_true_np = _to_numpy(Y_true)
    mean_before_np = _to_numpy(mean_before)
    mean_after_np = _to_numpy(mean_after)

    # Compute Wasserstein Distance
    wasserstein_before = wasserstein_distance(Y_true_np, mean_before_np)
    wasserstein_after = wasserstein_distance(Y_true_np, mean_after_np)

    print(f"📏 Wasserstein Distance Before Update: {wasserstein_before:.4f}")
    print(f"📏 Wasserstein Distance After Update:  {wasserstein_after:.4f}")

    # KDE plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(Y_true_np, label="Analytical P_combined", color="black", lw=2)
    sns.kdeplot(mean_before_np, label=f"{label_prefix} Before Update", color="blue", lw=2)
    sns.kdeplot(mean_after_np, label=f"{label_prefix} After Update", color="green", lw=2)
    plt.xlabel("P_combined Value")
    plt.ylabel("Density")
    plt.title("KDE of P_combined Distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[Y_true_np, mean_before_np, mean_after_np],
                palette=["black", "blue", "green"])
    plt.xticks(ticks=[0, 1, 2], labels=["Analytical", f"{label_prefix} Before", f"{label_prefix} After"])
    plt.title("Boxplot of P_combined Distributions")
    plt.ylabel("P_combined Value")
    plt.tight_layout()
    plt.show()


def _to_numpy(x):
    """
    Helper to convert Torch or NumPy input to flattened NumPy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().flatten()
    elif isinstance(x, np.ndarray):
        return x.flatten()
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
