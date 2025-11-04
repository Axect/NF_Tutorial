import torch
from torch import nn
import torch.distributions as dist


class MLP(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super(MLP, self).__init__()
        self.hparams = hparams
        self.device = device

        nodes = hparams["nodes"]
        layers = hparams["layers"]
        input_size = 1
        output_size = 1

        net = [nn.Linear(input_size, nodes), nn.GELU()]
        for _ in range(layers - 1):
            net.append(nn.Linear(nodes, nodes))
            net.append(nn.GELU())
        net.append(nn.Linear(nodes, output_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask_parity):
        """
        A coupling layer for normalizing flows.
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layers in the MLP.
            mask_parity (str): 'even' or 'odd' to determine which half of the input to mask.
        """
        super().__init__()

        assert input_dim % 2 == 0, "Input dimension must be even."

        self.dim1 = input_dim // 2
        self.dim2 = input_dim - self.dim1
        self.mask_parity = mask_parity

        def create_net(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.s_net = create_net(self.dim1, self.dim2)
        self.t_net = create_net(self.dim1, self.dim2)

    def forward(self, x):
        if self.mask_parity == "even":
            x1, x2 = x[..., : self.dim1], x[..., self.dim1 :]
        else:
            x2, x1 = x[..., : self.dim2], x[..., self.dim2 :]

        s = torch.tanh(self.s_net(x1))  # prevent blowup
        t = self.t_net(x1)

        y1 = x1
        y2 = x2 * torch.exp(s) + t  # Affine transformation

        if self.mask_parity == "even":
            y = torch.cat([y1, y2], dim=-1)
        else:
            y = torch.cat([y2, y1], dim=-1)

        log_det_J = s.sum(dim=-1)

        return y, log_det_J

    def inverse(self, y):
        if self.mask_parity == "even":
            y1, y2 = y[..., : self.dim1], y[..., self.dim1 :]
        else:
            y2, y1 = y[..., : self.dim2], y[..., self.dim2 :]

        x1 = y1

        s = torch.tanh(self.s_net(x1))
        t = self.t_net(x1)

        x2 = (y2 - t) * torch.exp(-s)  # Inverse affine transformation

        if self.mask_parity == "even":
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = torch.cat([x2, x1], dim=-1)

        log_det_J_inv = -s.sum(dim=-1)

        return x, log_det_J_inv


class NormalizingFlow(nn.Module):
    def __init__(self, hparams, device='cpu'):
        super().__init__()

        self.input_dim = hparams["input_dim"]
        self.hidden_dim = hparams["hidden_dim"]
        self.num_layers = hparams["num_layers"]
        self.device = device

        # Create a sequence of coupling layers with alternating masks
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            parity = "even" if i % 2 == 0 else "odd"
            self.layers.append(CouplingLayer(self.input_dim, self.hidden_dim, parity))

        # Base distribution: Standard normal
        self.register_buffer("base_mean", torch.zeros(self.input_dim))
        self.register_buffer("base_cov", torch.eye(self.input_dim))

    @property
    def base_dist(self):
        return dist.MultivariateNormal(self.base_mean, self.base_cov)

    def log_prob(self, x):
        """
        Compute the log probability of data point x under the normalizing flow model.

        log p_X(x) = log p_Z(z_0) + log|det(J(f^-1)(x))|
        """
        z = x
        total_log_det_inv = torch.zeros(x.shape[0], device=x.device)

        for layer in reversed(self.layers):
            z, log_det_inv = layer.inverse(z)
            total_log_det_inv += log_det_inv

        log_prob_z0 = self.base_dist.log_prob(z)
        log_prob_x = log_prob_z0 + total_log_det_inv
        return log_prob_x

    def forward(self, x):
        """
        Compute NLL = -log p_X(x)
        """
        log_prob_x = self.log_prob(x)
        nll = -torch.mean(log_prob_x)
        return nll

    def sample(self, num_samples):
        """
        Generate samples from the normalizing flow model.

        1. Sample z_0 from base distribution p_Z(z_0)
        2. x = f(z_0) through the flow layers
        """
        with torch.no_grad():
            z = self.base_dist.sample((num_samples,))

            for layer in self.layers:
                z, _ = layer.forward(z)

            x = z
        return x
