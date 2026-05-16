"""
Lag_Net with LoRA-adapted V_net for perturbation identification.

Instead of adding a separate delta_V_net to the output, LoRA modifies
V_net's internal weight matrices:

    W_effective = W_frozen + A @ B

where A is (d_out x rank) and B is (rank x d_in). Only A and B are
trainable. This lets the perturbation modify V_net's internal feature
computation rather than just adding an output-space correction.

Theoretically motivated: perturbations are low-dimensional modifications
to the potential energy landscape, so a low-rank update to the weight
matrices should suffice.
"""
import torch
import torch.nn as nn
from lag_caVAE.lag import Lag_Net


class LoRALinear(nn.Module):
    """Linear layer with frozen base weights and trainable low-rank residual.

    W_effective = W_frozen + (alpha/rank) * A @ B

    A: (d_out, rank) initialized from N(0, 1)
    B: (rank, d_in) initialized to zeros
    => initial output = W_frozen @ x + bias (no perturbation at init)
    """

    def __init__(self, base_linear, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weights
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(self.out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.in_features))

    def forward(self, x):
        # Base computation (frozen)
        out = nn.functional.linear(x, self.weight, self.bias)
        # LoRA residual (trainable)
        out = out + nn.functional.linear(x, self.lora_A @ self.lora_B) * self.scaling
        return out

    def get_effective_weight(self):
        """Return the effective weight matrix W + scaling * A @ B."""
        return self.weight + self.scaling * (self.lora_A @ self.lora_B)


def apply_lora_to_mlp(mlp, rank=4, alpha=1.0, layers='all'):
    """Wrap an MLP's linear layers with LoRA adapters.

    Args:
        mlp: An MLP instance with linear1, linear2, linear3.
        rank: LoRA rank (number of trainable dimensions per layer).
        alpha: LoRA scaling factor.
        layers: 'all' to wrap all 3 layers, or list like [1,2,3].

    Returns:
        The modified mlp (in-place).
    """
    if layers == 'all':
        layers = [1, 2, 3]

    for i in layers:
        attr = f'linear{i}'
        base_linear = getattr(mlp, attr)
        lora_linear = LoRALinear(base_linear, rank=rank, alpha=alpha)
        setattr(mlp, attr, lora_linear)

    return mlp


class Lag_Net_LoRA(Lag_Net):
    """
    Lag_Net where V_net has LoRA-adapted linear layers.

    The V_net is a standard MLP whose linear layers have been wrapped
    with LoRALinear. The forward pass is identical to Lag_Net — V_net
    directly produces V_total(cos_q, sin_q) with the perturbation
    baked into the weight matrices.

    Stores self.V_q for analysis compatibility with existing notebooks.
    """

    def __init__(self, q_dim=1, u_dim=1,
                 g_net=None, M_net=None, V_net=None,
                 dyna_model='lag'):
        super().__init__(
            q_dim=q_dim, u_dim=u_dim,
            g_net=g_net, M_net=M_net, V_net=V_net,
            dyna_model=dyna_model,
        )

    def forward(self, t, x, **kwargs):
        if self.dyna_model != 'lag':
            return super().forward(t, x)

        cos_q, sin_q, q_dot, u = x.split(
            [self.q_dim, self.q_dim, self.q_dim, self.u_dim], dim=1
        )
        cos_q_sin_q = torch.cat((cos_q, sin_q), dim=1)
        if not cos_q_sin_q.requires_grad:
            cos_q_sin_q.requires_grad_(True)
        d_cos_q = -sin_q * q_dot
        d_sin_q = cos_q * q_dot

        self.M_q = self.M_net(cos_q_sin_q)
        self.V_q = self.V_net(cos_q_sin_q)

        dV = torch.autograd.grad(
            self.V_q.sum(), cos_q_sin_q, create_graph=True
        )[0]
        dV_dq = (dV[:, 0:self.q_dim] * (-sin_q)
                 + dV[:, self.q_dim:2*self.q_dim] * cos_q)

        if self.q_dim == 1:
            dM = torch.autograd.grad(
                self.M_q.sum(), cos_q_sin_q, create_graph=True
            )[0]
            dM_dq = (dM[:, 0:self.q_dim] * (-sin_q)
                     + dM[:, self.q_dim:2*self.q_dim] * cos_q)
            d_q_dot = (-0.5 * q_dot * q_dot * dM_dq
                       - dV_dq
                       + self.g_net(cos_q_sin_q) * u)
            d_q_dot = d_q_dot / self.M_q
        else:
            raise NotImplementedError("LoRA V_net only supports q_dim=1 for now")

        return torch.cat(
            [d_cos_q, d_sin_q, d_q_dot, torch.zeros_like(u)], dim=1
        )
