import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from metric_semantic_predicate.models.bayesian_combined_bnn import BayesianNNcombined
# Extendable: You can register other models here
# from metric_semantic_predicate.models.metric_model import MetricModel
# from metric_semantic_predicate.models.semantic_model import SemanticModel

def get_training_device():
    """
    Returns 'cuda' if GPU is available, else 'cpu'
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_and_target_cols(config_path=None, task="combined"):
    """
    Returns feature and target columns based on task type or config file

    Parameters:
    - config_path (str): Path to JSON config (optional)
    - task (str): One of ['combined', 'metric', 'semantic', 'predicate']

    Returns:
    - feature_cols (list)
    - target_cols (list)
    """
    if config_path:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return cfg["feature_cols"], cfg["target_cols"]

    if task == "combined":
        feature_cols = ["metric", "encoded_predicate", "encoded_semantic", "width", "height", "depth"]
        target_cols = ["P_combined"]
    elif task == "predicate":
        feature_cols = ["encoded_predicate", "width", "height", "depth"]
        target_cols = ["theta0", "phi0", "kappa"]
    elif task == "semantic":
        feature_cols = ["encoded_semantic", "width", "height", "depth"]
        target_cols = ["mu_x", "mu_y", "mu_z", "sigma_s"]
    elif task == "metric":
        feature_cols = ["metric", "width", "height", "depth"]
        target_cols = ["x0", "y0", "z0", "d0", "sigma_m"]
    else:
        raise ValueError(f"Unsupported task type: {task}")

    return feature_cols, target_cols


def prepare_tensor_data(X, Y, device):
    """
    Converts numpy arrays to torch tensors and moves to device

    Parameters:
    - X (np.ndarray): Features
    - Y (np.ndarray): Targets
    - device (str or torch.device)

    Returns:
    - X_t, Y_t (torch.Tensor)
    """
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
    return X_t, Y_t


def create_dataloader(X, Y, batch_size=64, shuffle=True, device="cpu"):
    """
    Returns a PyTorch DataLoader

    Parameters:
    - X (np.ndarray or torch.Tensor)
    - Y (np.ndarray or torch.Tensor)
    - batch_size (int)
    - shuffle (bool)
    - device (str or torch.device)

    Returns:
    - DataLoader instance
    """
    X_t, Y_t = prepare_tensor_data(X, Y, device)
    dataset = TensorDataset(X_t, Y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def initialize_bnn_model(model_type="combined", input_dim=None, hidden_dim=32, output_dim=1, device="cpu"):
    """
    Returns a Bayesian model instance on the selected device.

    Parameters:
    - model_type (str): One of ['combined', 'metric', 'semantic', 'predicate']
    - input_dim (int)
    - hidden_dim (int)
    - output_dim (int)
    - device (str or torch.device)

    Returns:
    - model instance (e.g., BayesianNNcombined)
    """
    if model_type == "combined":
        return BayesianNNcombined(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    # elif model_type == "metric":
    #     return MetricModel(input_dim, hidden_dim, output_dim).to(device)
    # elif model_type == "semantic":
    #     return SemanticModel(input_dim, hidden_dim, output_dim).to(device)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")
