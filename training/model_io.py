import os
import torch
import pyro

def save_model(model_name, save_dir="data/models"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, model_name + ".pt")
    torch.save(pyro.get_param_store().get_state(), path)
    print(f"✅ Saved model to {path}")

def load_model(model_name, save_dir="data/models"):
    path = os.path.join(save_dir, model_name + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found at {path}")
    pyro.clear_param_store()
    pyro.get_param_store().set_state(torch.load(path))
    print(f"✅ Loaded model from {path}")
