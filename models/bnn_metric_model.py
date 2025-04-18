import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime


class BayesianNN_metric(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x, w1, b1, w2, b2):
        w1, b1, w2, b2 = w1.to(x.device), b1.to(x.device), w2.to(x.device), b2.to(x.device)
        hidden = torch.relu(torch.matmul(x, w1) + b1)
        return torch.matmul(hidden, w2) + b2


def model_bnn_metric(x, y, net):
    w1 = pyro.sample("metric_w1", dist.Normal(torch.zeros(net.input_dim, net.hidden_dim), 0.1 * torch.ones(net.input_dim, net.hidden_dim)).to_event(2))
    b1 = pyro.sample("metric_b1", dist.Normal(torch.zeros(net.hidden_dim), 0.1 * torch.ones(net.hidden_dim)).to_event(1))
    w2 = pyro.sample("metric_w2", dist.Normal(torch.zeros(net.hidden_dim, net.output_dim), 0.1 * torch.ones(net.hidden_dim, net.output_dim)).to_event(2))
    b2 = pyro.sample("metric_b2", dist.Normal(torch.zeros(net.output_dim), 0.1 * torch.ones(net.output_dim)).to_event(1))
    sigma = pyro.sample("metric_sigma", dist.Exponential(0.1 * torch.ones(net.output_dim)).to_event(1))
    pred = net.forward(x, w1, b1, w2, b2)
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(pred, sigma).to_event(1), obs=y)


def guide_bnn_metric(x, y, net):
    w1_scale = torch.clamp(torch.nn.functional.softplus(
        pyro.param("metric_w1_scale", 0.1 * torch.ones(net.input_dim, net.hidden_dim))), min=1e-3)
    b1_scale = torch.clamp(torch.nn.functional.softplus(
        pyro.param("metric_b1_scale", 0.1 * torch.ones(net.hidden_dim))), min=1e-3)
    w2_scale = torch.clamp(torch.nn.functional.softplus(
        pyro.param("metric_w2_scale", 0.1 * torch.ones(net.hidden_dim, net.output_dim))), min=1e-3)
    b2_scale = torch.clamp(torch.nn.functional.softplus(
        pyro.param("metric_b2_scale", 0.1 * torch.ones(net.output_dim))), min=1e-3)
    sigma_loc = pyro.param("metric_sigma_loc", 0.1 * torch.ones(net.output_dim), constraint=dist.constraints.positive)

    pyro.sample("metric_w1", dist.Normal(torch.zeros_like(w1_scale), w1_scale).to_event(2))
    pyro.sample("metric_b1", dist.Normal(torch.zeros_like(b1_scale), b1_scale).to_event(1))
    pyro.sample("metric_w2", dist.Normal(torch.zeros_like(w2_scale), w2_scale).to_event(2))
    pyro.sample("metric_b2", dist.Normal(torch.zeros_like(b2_scale), b2_scale).to_event(1))
    pyro.sample("metric_sigma", dist.Exponential(sigma_loc).to_event(1))


def train_bnn_model_metric(X, Y, input_dim, output_dim, hidden_dim=32, num_steps=20000, lr=0.0005, device="cpu", model_name="bnn_metric", writer=None):
    pyro.clear_param_store()
    net = BayesianNN_metric(input_dim, hidden_dim, output_dim).to(device)
    X_t, Y_t = X.to(device), Y.to(device)

    optim = Adam({"lr": lr})
    svi = SVI(lambda x, y: model_bnn_metric(x, y, net),
              lambda x, y: guide_bnn_metric(x, y, net),
              optim, Trace_ELBO())

    internal_writer = False
    if writer is None:
        writer = SummaryWriter(log_dir=f"runs/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        internal_writer = True

    for step in range(num_steps):
        loss = svi.step(X_t, Y_t)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        writer.add_scalar("Loss/train", loss, step)

        if step % 500 == 0:
            print(f"[METRIC BNN] Step {step}, loss={loss:.4f}")

    if internal_writer:
        writer.close()

    return svi, net


def bnn_predict_metric(X, net, n_samples=100):
    net.eval()
    samples = []

    for _ in range(n_samples):
        w1 = pyro.sample("metric_w1", dist.Normal(pyro.param("metric_w1_scale"), 0.1).to_event(2))
        b1 = pyro.sample("metric_b1", dist.Normal(pyro.param("metric_b1_scale"), 0.1).to_event(1))
        w2 = pyro.sample("metric_w2", dist.Normal(pyro.param("metric_w2_scale"), 0.1).to_event(2))
        b2 = pyro.sample("metric_b2", dist.Normal(pyro.param("metric_b2_scale"), 0.1).to_event(1))
        with torch.no_grad():
            preds = net.forward(X.to(w1.device), w1, b1, w2, b2)
            samples.append(preds.cpu().numpy())

    samples = np.stack(samples)
    return samples.mean(axis=0), samples.std(axis=0)
