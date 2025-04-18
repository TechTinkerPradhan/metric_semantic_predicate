import torch
saved = torch.load("data/models/bnn_metric_after_update.pt", map_location="cpu")
print(saved.keys())
