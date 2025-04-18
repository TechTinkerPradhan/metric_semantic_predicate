import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime


class BayesianNNSem(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x, w1, b1, w2, b2):
        w1, b1, w2, b2 = w1.to(x.device), b1.to(x.device), w2.to(x.device), b2.to(x.device)
        hidden = F.relu(torch.matmul(x, w1) + b1)
        return torch.matmul(hidden, w2) + b2

def model_bnn_sem(x, y, net):
    device = x.device
    w1 = pyro.sample("sem_w1", dist.Normal(torch.zeros(net.input_dim, net.hidden_dim, device=device), 0.1 * torch.ones(net.input_dim, net.hidden_dim, device=device)).to_event(2))
    b1 = pyro.sample("sem_b1", dist.Normal(torch.zeros(net.hidden_dim, device=device), 0.1 * torch.ones(net.hidden_dim, device=device)).to_event(1))
    w2 = pyro.sample("sem_w2", dist.Normal(torch.zeros(net.hidden_dim, net.output_dim, device=device), 0.1 * torch.ones(net.hidden_dim, net.output_dim, device=device)).to_event(2))
    b2 = pyro.sample("sem_b2", dist.Normal(torch.zeros(net.output_dim, device=device), 0.1 * torch.ones(net.output_dim, device=device)).to_event(1))
    sigma = pyro.sample("sem_sigma", dist.Exponential(0.1 * torch.ones(net.output_dim, device=device)).to_event(1))

    pred = net.forward(x, w1, b1, w2, b2)

    # 🚨 Make sure y is also on same device
    y = y.to(device)

    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(pred, sigma).to_event(1), obs=y)


def guide_bnn_sem(x, y, net):
    device = x.device

    w1_loc = pyro.param("sem_w1_loc", 0.1 * torch.randn(net.input_dim, net.hidden_dim, device=device))
    w1_scale = pyro.param("sem_w1_scale", 0.1 * torch.ones(net.input_dim, net.hidden_dim, device=device), constraint=dist.constraints.positive)

    b1_loc = pyro.param("sem_b1_loc", 0.1 * torch.randn(net.hidden_dim, device=device))
    b1_scale = pyro.param("sem_b1_scale", 0.1 * torch.ones(net.hidden_dim, device=device), constraint=dist.constraints.positive)

    w2_loc = pyro.param("sem_w2_loc", 0.1 * torch.randn(net.hidden_dim, net.output_dim, device=device))
    w2_scale = pyro.param("sem_w2_scale", 0.1 * torch.ones(net.hidden_dim, net.output_dim, device=device), constraint=dist.constraints.positive)

    b2_loc = pyro.param("sem_b2_loc", 0.1 * torch.randn(net.output_dim, device=device))
    b2_scale = pyro.param("sem_b2_scale", 0.1 * torch.ones(net.output_dim, device=device), constraint=dist.constraints.positive)

    sigma_loc = pyro.param("sem_sigma_loc", 0.1 * torch.ones(net.output_dim, device=device), constraint=dist.constraints.positive)

    pyro.sample("sem_w1", dist.Normal(w1_loc, w1_scale).to_event(2))
    pyro.sample("sem_b1", dist.Normal(b1_loc, b1_scale).to_event(1))
    pyro.sample("sem_w2", dist.Normal(w2_loc, w2_scale).to_event(2))
    pyro.sample("sem_b2", dist.Normal(b2_loc, b2_scale).to_event(1))
    pyro.sample("sem_sigma", dist.Exponential(sigma_loc).to_event(1))


def train_bnn_sem_model(X, Y, input_dim, output_dim, hidden_dim=32, num_steps=2000, lr=0.01, writer=None, X_val=None, Y_val=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BayesianNNSem(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    X_t = X.clone().detach().float().to(device)
    Y_t = Y.clone().detach().float().to(device)


    optim = Adam({"lr": lr})
    svi = SVI(
        model=lambda x, y: model_bnn_sem(x, y, net),
        guide=lambda x, y: guide_bnn_sem(x, y, net),
        optim=optim,
        loss=Trace_ELBO()
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if writer is None:
        writer = SummaryWriter(log_dir=f"runs/semantic_bnn_{timestamp}")

    for step in range(num_steps):
        loss = svi.step(X_t, Y_t)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        if writer:
            writer.add_scalar("Loss/train", loss, step)

        if step % 100 == 0 and X_val is not None and Y_val is not None:
            with torch.no_grad():
                val_loss = svi.evaluate_loss(X_val.to(device), Y_val.to(device))
                writer.add_scalar("Loss/val", val_loss, step)

        if step % 200 == 0:
            print(f"[SEMANTIC BNN] Step {step}, Loss={loss:.4f}")

    writer.close()
    print("✅ Semantic BNN training complete.")
    return svi, net

def bnn_predict_sem(X, net, n_samples=50):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    preds = []
    for _ in range(n_samples):
        w1 = dist.Normal(pyro.param("sem_w1_loc"), pyro.param("sem_w1_scale")).sample()
        b1 = dist.Normal(pyro.param("sem_b1_loc"), pyro.param("sem_b1_scale")).sample()
        w2 = dist.Normal(pyro.param("sem_w2_loc"), pyro.param("sem_w2_scale")).sample()
        b2 = dist.Normal(pyro.param("sem_b2_loc"), pyro.param("sem_b2_scale")).sample()
        with torch.no_grad():
            pred = net(X, w1, b1, w2, b2)
            preds.append(pred.cpu().numpy())
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)
