"""
Lag_Net with additive ΔV_net for perturbation identification.

V_total(cos_q, sin_q) = V_net_frozen(cos_q, sin_q) + delta_V_net(cos_q, sin_q)

The gradient dV_total/dq flows through both networks via autograd,
so the Euler-Lagrange equation automatically picks up the perturbation.
"""
import torch
from lag_caVAE.lag import Lag_Net


class Lag_Net_DeltaV(Lag_Net):
    """
    Extends Lag_Net with an additive delta_V_net.

    Everything is identical to Lag_Net except that in forward(),
    V_q = V_net(cos_q_sin_q) + delta_V_net(cos_q_sin_q)
    before computing dV/dq via autograd.
    """

    def __init__(self, q_dim=1, u_dim=1,
                 g_net=None, M_net=None, V_net=None,
                 delta_V_net=None, dyna_model='lag'):
        super().__init__(
            q_dim=q_dim, u_dim=u_dim,
            g_net=g_net, M_net=M_net, V_net=V_net,
            dyna_model=dyna_model,
        )
        self.delta_V_net = delta_V_net

    def forward(self, t, x, **kwargs):
        # For non-lagrangian models, fall back to parent
        if self.dyna_model != 'lag':
            return super().forward(t, x)

        cos_q, sin_q, q_dot, u = x.split(
            [self.q_dim, self.q_dim, self.q_dim, self.u_dim], dim=1
        )
        cos_q_sin_q = torch.cat((cos_q, sin_q), dim=1)
        # Ensure requires_grad for autograd.grad through V_net/M_net/delta_V_net.
        # When the encoder is frozen, ODE state tensors may not carry gradients,
        # but we still need dV/d(cos_q, sin_q) for the Euler-Lagrange equation.
        if not cos_q_sin_q.requires_grad:
            cos_q_sin_q.requires_grad_(True)
        d_cos_q = -sin_q * q_dot
        d_sin_q = cos_q * q_dot

        # --- Potential: V_total = V_net + delta_V_net ---
        self.M_q = self.M_net(cos_q_sin_q)
        self.V_q = self.V_net(cos_q_sin_q)
        if self.delta_V_net is not None:
            self.delta_V_q = self.delta_V_net(cos_q_sin_q)
            V_total = self.V_q + self.delta_V_q
        else:
            V_total = self.V_q

        # dV/dq via chain rule through (cos_q, sin_q)
        dV = torch.autograd.grad(
            V_total.sum(), cos_q_sin_q, create_graph=True
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
            # Multi-DOF case (same as parent Lag_Net)
            dM_dt = torch.zeros_like(self.M_q)
            for row_ind in range(self.q_dim):
                for col_ind in range(self.q_dim):
                    dM = torch.autograd.grad(
                        self.M_q[:, row_ind, col_ind].sum(),
                        cos_q_sin_q, create_graph=True
                    )[0]
                    dM_dt[:, row_ind, col_ind] = (
                        dM * torch.cat((-sin_q * q_dot, cos_q * q_dot), dim=1)
                    ).sum(-1)
            q_dot_M_q_dot = torch.matmul(
                torch.unsqueeze(q_dot, 1),
                torch.matmul(self.M_q, torch.unsqueeze(q_dot, 2))
            )
            d_q_dot_M_q_dot = torch.autograd.grad(
                q_dot_M_q_dot.sum(), cos_q_sin_q, create_graph=True
            )[0]
            d_q_dot_M_q_dot_dq = (
                d_q_dot_M_q_dot[:, 0:self.q_dim] * (-sin_q)
                + d_q_dot_M_q_dot[:, self.q_dim:2*self.q_dim] * cos_q
            )
            temp = (
                -torch.matmul(dM_dt, q_dot[:, :, None])
                + 0.5 * d_q_dot_M_q_dot_dq[:, :, None]
                - dV_dq[:, :, None]
                + torch.matmul(self.g_net(cos_q_sin_q), u[:, :, None])
            )
            d_q_dot = torch.squeeze(
                torch.matmul(torch.inverse(self.M_q), temp), 2
            )

        return torch.cat(
            [d_cos_q, d_sin_q, d_q_dot, torch.zeros_like(u)], dim=1
        )
