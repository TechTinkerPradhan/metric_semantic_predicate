import os
import torch
import pyro

def save_model(model_name, save_dir="data/models"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, model_name + ".pt")
    # Save only the param store state for compatibility
    torch.save(pyro.get_param_store().get_state(), path)
    print(f"✅ Saved model to {path}")

def load_model(model_name, save_dir="data/models"):
    path = os.path.join(save_dir, model_name + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found at {path}")
    map_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyro.get_param_store().set_state(torch.load(path, map_location=map_device))
    print(f"✅ Loaded model from {path}")
