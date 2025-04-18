import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime

class BayesianNNPred(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x, w1, b1, w2, b2):
        hidden = torch.relu(torch.matmul(x, w1) + b1)
        out = torch.matmul(hidden, w2) + b2
        return out

def model_bnn_pred(x, y, net):
    w1 = pyro.sample("pred_w1", dist.Normal(torch.zeros(net.input_dim, net.hidden_dim), torch.ones(net.input_dim, net.hidden_dim)).to_event(2))
    b1 = pyro.sample("pred_b1", dist.Normal(torch.zeros(net.hidden_dim), torch.ones(net.hidden_dim)).to_event(1))
    w2 = pyro.sample("pred_w2", dist.Normal(torch.zeros(net.hidden_dim, net.output_dim), torch.ones(net.hidden_dim, net.output_dim)).to_event(2))
    b2 = pyro.sample("pred_b2", dist.Normal(torch.zeros(net.output_dim), torch.ones(net.output_dim)).to_event(1))
    sigma = pyro.sample("pred_sigma", dist.Exponential(torch.ones(net.output_dim)).to_event(1))
    pred = net.forward(x, w1, b1, w2, b2)
    with pyro.plate("data_pred", x.shape[0]):
        pyro.sample("pred_obs", dist.Normal(pred, sigma).to_event(1), obs=y)

def guide_bnn_pred(x, y, net):
    w1_loc = pyro.param("pred_w1_loc", torch.zeros(net.input_dim, net.hidden_dim))
    w1_scale = pyro.param("pred_w1_scale", torch.ones(net.input_dim, net.hidden_dim), constraint=dist.constraints.positive)
    b1_loc = pyro.param("pred_b1_loc", torch.zeros(net.hidden_dim))
    b1_scale = pyro.param("pred_b1_scale", torch.ones(net.hidden_dim), constraint=dist.constraints.positive)
    w2_loc = pyro.param("pred_w2_loc", torch.zeros(net.hidden_dim, net.output_dim))
    w2_scale = pyro.param("pred_w2_scale", torch.ones(net.hidden_dim, net.output_dim), constraint=dist.constraints.positive)
    b2_loc = pyro.param("pred_b2_loc", torch.zeros(net.output_dim))
    b2_scale = pyro.param("pred_b2_scale", torch.ones(net.output_dim), constraint=dist.constraints.positive)
    sigma_loc = pyro.param("pred_sigma_loc", torch.ones(net.output_dim), constraint=dist.constraints.positive)

    pyro.sample("pred_w1", dist.Normal(w1_loc, w1_scale).to_event(2))
    pyro.sample("pred_b1", dist.Normal(b1_loc, b1_scale).to_event(1))
    pyro.sample("pred_w2", dist.Normal(w2_loc, w2_scale).to_event(2))
    pyro.sample("pred_b2", dist.Normal(b2_loc, b2_scale).to_event(1))
    pyro.sample("pred_sigma", dist.Exponential(sigma_loc).to_event(1))

def bnn_predict_predicate(X, net, n_samples=50):
    if not isinstance(X, torch.Tensor):
        X_t = torch.tensor(X, dtype=torch.float32)
    else:
        X_t = X

    preds = []
    for _ in range(n_samples):
        w1 = dist.Normal(pyro.param("pred_w1_loc"), pyro.param("pred_w1_scale")).sample()
        b1 = dist.Normal(pyro.param("pred_b1_loc"), pyro.param("pred_b1_scale")).sample()
        w2 = dist.Normal(pyro.param("pred_w2_loc"), pyro.param("pred_w2_scale")).sample()
        b2 = dist.Normal(pyro.param("pred_b2_loc"), pyro.param("pred_b2_scale")).sample()
        net.eval()
        out = net.forward(X_t, w1, b1, w2, b2)
        preds.append(out.detach().cpu().numpy())

    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def train_bnn_pred_model(X, Y, input_dim, output_dim=2, hidden_dim=32, num_steps=2000, lr=0.01, writer=None, X_val=None, Y_val=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BayesianNNPred(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)

    optim = Adam({"lr": lr})
    svi = SVI(lambda x, y: model_bnn_pred(x, y, net),
              lambda x, y: guide_bnn_pred(x, y, net),
              optim, Trace_ELBO())

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if writer is None:
        writer = SummaryWriter(log_dir=f"runs/predicate_bnn_{timestamp}")

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
            print(f"[PREDICATE BNN] Step {step}, Loss={loss:.4f}")

    writer.close()
    print("✅ Predicate BNN training complete.")
    return svi, net