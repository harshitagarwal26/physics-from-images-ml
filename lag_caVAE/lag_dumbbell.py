"""
Lag_Net for the rigid-dumbbell system: 1 Euclidean + 2 angular generalized
coordinates (r, phi, theta).

State vector for the ODE:
    [ r, cos_phi, sin_phi, cos_theta, sin_theta,
      r_dot, phi_dot, theta_dot, u ]                  length = 8 + u_dim

V_net, M_net, g_net input: (r, cos_theta, sin_theta)
    ^ phi is deliberately excluded to enforce the rotational symmetry of a
      central gravity field as a hard inductive bias (V is phi-invariant).

Equations of motion, obtained from Euler-Lagrange with L = T - V:
    M q_ddot = - dM/dt q_dot + 0.5 d/dq (q_dot^T M q_dot) - dV/dq + g u
with q = (r, phi, theta). Because phi is absent from the inputs of V, M, g,
the phi-components of dV/dq, dM/dq, d(q_dot^T M q_dot)/dq are identically 0.
"""

import torch


class Lag_Net_Dumbbell(torch.nn.Module):
    def __init__(self, g_net=None, M_net=None, V_net=None, u_dim=1):
        super().__init__()
        self.g_net = g_net
        self.M_net = M_net
        self.V_net = V_net
        self.u_dim = u_dim

    def forward(self, t, x):
        # Split the augmented ODE state.
        r, cos_phi, sin_phi, cos_theta, sin_theta, \
            r_dot, phi_dot, theta_dot, u = x.split(
                [1, 1, 1, 1, 1, 1, 1, 1, self.u_dim], dim=1,
            )

        # Kinematic time derivatives.
        dr = r_dot
        dcos_phi = -sin_phi * phi_dot
        dsin_phi = cos_phi * phi_dot
        dcos_theta = -sin_theta * theta_dot
        dsin_theta = cos_theta * theta_dot

        # The "geometric position" input to V, M, g — phi deliberately omitted.
        geom = torch.cat([r, cos_theta, sin_theta], dim=1)

        self.M_q = self.M_net(geom)       # (bs, 3, 3)
        self.V_q = self.V_net(geom)       # (bs, 1)

        # dV/dq via chain rule: dV/d(geom) * d(geom)/dq
        dV = torch.autograd.grad(self.V_q.sum(), geom, create_graph=True)[0]
        dV_dr, dV_dcos_t, dV_dsin_t = dV.split([1, 1, 1], dim=1)
        dV_dtheta = dV_dcos_t * (-sin_theta) + dV_dsin_t * cos_theta
        # dV/dphi = 0 by construction.
        dV_dq = torch.cat([dV_dr, torch.zeros_like(dV_dr), dV_dtheta], dim=1)

        # dM/dt: chain rule through (r, cos_theta, sin_theta).
        # d/dt [r] = r_dot,  d/dt [cos_theta] = -sin_theta*theta_dot,
        # d/dt [sin_theta] =  cos_theta*theta_dot.
        t_deriv_inputs = torch.cat(
            [r_dot, -sin_theta * theta_dot, cos_theta * theta_dot], dim=1,
        )
        dM_dt = torch.zeros_like(self.M_q)
        for i in range(3):
            for j in range(3):
                dMij = torch.autograd.grad(
                    self.M_q[:, i, j].sum(), geom, create_graph=True,
                )[0]                                     # (bs, 3)
                dM_dt[:, i, j] = (dMij * t_deriv_inputs).sum(-1)

        # d/dq of ( q_dot^T M q_dot )
        q_dot = torch.cat([r_dot, phi_dot, theta_dot], dim=1)              # (bs, 3)
        q_dot_M_q_dot = torch.matmul(
            q_dot[:, None, :],
            torch.matmul(self.M_q, q_dot[:, :, None]),
        )                                                                  # (bs, 1, 1)
        dE = torch.autograd.grad(
            q_dot_M_q_dot.sum(), geom, create_graph=True,
        )[0]                                                               # (bs, 3)
        dE_dr, dE_dcos_t, dE_dsin_t = dE.split([1, 1, 1], dim=1)
        dE_dtheta = dE_dcos_t * (-sin_theta) + dE_dsin_t * cos_theta
        dE_dq = torch.cat([dE_dr, torch.zeros_like(dE_dr), dE_dtheta], dim=1)

        # Assemble RHS and solve M q_ddot = rhs
        rhs = (
            -torch.matmul(dM_dt, q_dot[:, :, None])
            + 0.5 * dE_dq[:, :, None]
            - dV_dq[:, :, None]
        )
        if self.u_dim > 0:
            g = self.g_net(geom)                                            # (bs, 3, u_dim)
            rhs = rhs + torch.matmul(g, u[:, :, None])

        q_ddot = torch.squeeze(
            torch.matmul(torch.inverse(self.M_q), rhs), dim=2,
        )                                                                   # (bs, 3)
        d_r_dot, d_phi_dot, d_theta_dot = q_ddot.split([1, 1, 1], dim=1)

        return torch.cat(
            [dr, dcos_phi, dsin_phi, dcos_theta, dsin_theta,
             d_r_dot, d_phi_dot, d_theta_dot,
             torch.zeros_like(u)],
            dim=1,
        )
