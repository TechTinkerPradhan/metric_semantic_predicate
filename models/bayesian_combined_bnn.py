import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np

class BayesianNNcombined(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        self.input_dim = input_dim  
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x, w1, b1, w2, b2):
        hidden = F.relu(torch.matmul(x, w1) + b1)
        return torch.matmul(hidden, w2) + b2


def model_bnn_combined(x, y, net):
    w1 = pyro.sample("w1", dist.Normal(torch.zeros(net.input_dim, net.hidden_dim), torch.ones(net.input_dim, net.hidden_dim)).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(torch.zeros(net.hidden_dim), torch.ones(net.hidden_dim)).to_event(1))
    w2 = pyro.sample("w2", dist.Normal(torch.zeros(net.hidden_dim, net.output_dim), torch.ones(net.hidden_dim, net.output_dim)).to_event(2))
    b2 = pyro.sample("b2", dist.Normal(torch.zeros(net.output_dim), torch.ones(net.output_dim)).to_event(1))
    sigma = pyro.sample("sigma", dist.Exponential(torch.ones(net.output_dim)).to_event(1))
    pred = net(x, w1, b1, w2, b2)
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(pred, sigma).to_event(1), obs=y)


def guide_bnn_combined(x, y, net):
    w1_loc = pyro.param("combined_w1_loc", torch.zeros(net.input_dim, net.hidden_dim))
    w1_scale = pyro.param("combined_w1_scale", torch.ones(net.input_dim, net.hidden_dim), constraint=dist.constraints.positive)
    b1_loc = pyro.param("combined_b1_loc", torch.zeros(net.hidden_dim))
    b1_scale = pyro.param("combined_b1_scale", torch.ones(net.hidden_dim), constraint=dist.constraints.positive)
    w2_loc = pyro.param("combined_w2_loc", torch.zeros(net.hidden_dim, net.output_dim))
    w2_scale = pyro.param("combined_w2_scale", torch.ones(net.hidden_dim, net.output_dim), constraint=dist.constraints.positive)
    b2_loc = pyro.param("combined_b2_loc", torch.zeros(net.output_dim))
    b2_scale = pyro.param("combined_b2_scale", torch.ones(net.output_dim), constraint=dist.constraints.positive)
    sigma_loc = pyro.param("combined_sigma_loc", torch.ones(net.output_dim), constraint=dist.constraints.positive)

    pyro.sample("w1", dist.Normal(w1_loc, w1_scale).to_event(2))
    pyro.sample("b1", dist.Normal(b1_loc, b1_scale).to_event(1))
    pyro.sample("w2", dist.Normal(w2_loc, w2_scale).to_event(2))
    pyro.sample("b2", dist.Normal(b2_loc, b2_scale).to_event(1))
    pyro.sample("sigma", dist.Exponential(sigma_loc).to_event(1))


def train_bnn_combined(X, Y, input_dim, hidden_dim=32, num_steps=2000, lr=0.01, device="cpu", model_name="bnn"):
    net = BayesianNNcombined(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optim = Adam({"lr": lr})
    svi = SVI(model=lambda x, y: model_bnn_combined(x, y, net),
              guide=lambda x, y: guide_bnn_combined(x, y, net),
              optim=optim,
              loss=Trace_ELBO())

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{model_name}_{timestamp}")

    for step in range(num_steps):
        loss = svi.step(X, Y)
        writer.add_scalar("Loss/train", loss, step)
        if step % 200 == 0:
            print(f"[BNN] Step {step}, Loss={loss:.4f}")

    writer.close()
    print("BNN Training Complete.")
    return svi, net


def bnn_predict(X, net, n_samples=50):
    preds = []
    for _ in range(n_samples):
        w1 = dist.Normal(pyro.param("combined_w1_loc"), pyro.param("combined_w1_scale")).sample()
        b1 = dist.Normal(pyro.param("combined_b1_loc"), pyro.param("combined_b1_scale")).sample()
        w2 = dist.Normal(pyro.param("combined_w2_loc"), pyro.param("combined_w2_scale")).sample()
        b2 = dist.Normal(pyro.param("combined_b2_loc"), pyro.param("combined_b2_scale")).sample()
        out = net(X, w1, b1, w2, b2)
        preds.append(out.detach().cpu().numpy())

    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0).squeeze(), preds.std(axis=0).squeeze()
