"""
Coordinate-aware VAE encoder/decoder for the rigid-dumbbell system.

Latent structure:
    r        Euclidean scalar in normalized image units  ( ~ [-1, 1] )
    phi      S^1   (orbital angle)           encoded as (cos phi, sin phi)
    theta    S^1   (attitude, body vs local-radial) encoded as (cos theta, sin theta)

Encoder (two-stage, following the cartpole caVAE):
    stage 1: MLP_Encoder on the full image  ->  (r, cos phi, sin phi) + logvars
    stage 2: warp the image so the CoM is at the center and the local radial
             points along +x (removes orbital angle)  ->  MLP_Encoder on the
             warped image -> (cos theta, sin theta) + logvar.

Decoder (compositional):
    a canonical body sprite (two masses, horizontal, centered) is warped by
    (r, phi+theta) to place it in world frame; a canonical planet sprite
    stays at the image center. Sum of the two is the reconstruction.

Neither module samples distributions; the enclosing Lightning module builds
Normal / VonMisesFisher from the outputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lag_caVAE.nn_models import MLP_Encoder, MLP_Decoder


def make_synthetic_body_canon(img_size=32, world_bound=5.0, ell=1.0,
                              particle_radius_px=2.0,
                              head_brightness=1.0, tail_brightness=0.6):
    """Synthetic dumbbell sprite in the canonical frame: body axis along +x,
    CoM at image center. Brightnesses (head=1.0, tail=0.6) match the env
    renderer in myenv/dumbbell.py. Used to pre-initialize the decoder so the
    encoder/decoder bootstrap deadlock breaks at epoch 0."""
    img = np.zeros((img_size, img_size), dtype=np.float32)
    ell_norm = ell / world_bound
    px_offset = ell_norm * img_size / 2.0
    cx = (img_size - 1) / 2.0
    cy = cx

    def _draw(px, py, r_px, brightness):
        for row in range(max(0, int(py - r_px - 1)), min(img_size, int(py + r_px + 2))):
            for col in range(max(0, int(px - r_px - 1)), min(img_size, int(px + r_px + 2))):
                dist = np.sqrt((col - px) ** 2 + (row - py) ** 2)
                if dist <= r_px - 0.5:
                    img[row, col] = max(img[row, col], brightness)
                elif dist <= r_px + 0.5:
                    alpha = r_px + 0.5 - dist
                    img[row, col] = max(img[row, col], brightness * alpha)

    _draw(cx + px_offset, cy, particle_radius_px, head_brightness)
    _draw(cx - px_offset, cy, particle_radius_px, tail_brightness)
    return img


def get_theta(cos, sin, x, y, bs, device, dtype):
    """Forward-warp affine: output(x_out, y_out) samples input at
    ( cos*x_out + sin*y_out + x ,  -sin*x_out + cos*y_out + y ).
    Used by the encoder to de-translate / de-rotate."""
    theta = torch.zeros([bs, 2, 3], dtype=dtype, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = sin
    theta[:, 0, 2] = x
    theta[:, 1, 0] = -sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = y
    return theta


def get_theta_inv(cos, sin, x, y, bs, device, dtype):
    """Inverse-warp affine for the decoder: places a body-frame sprite at
    (x, y) world and rotates it by the angle whose (cos, sin) is given."""
    theta = torch.zeros([bs, 2, 3], dtype=dtype, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = -x * cos + y * sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = -x * sin - y * cos
    return theta


class DumbbellEncoder(nn.Module):
    """
    Input:  (bs, 1, 32, 32)
    Output dict with keys:
        r_mean, r_var               (bs, 1)    Normal posterior over r
        phi_vec, phi_concentration  (bs, 2), (bs, 1)  vMF posterior over phi
        theta_vec, theta_concentration  (bs, 2), (bs, 1)  vMF posterior over theta
    """
    def __init__(self, img_size=32, hidden=300, nonlinearity='elu'):
        super().__init__()
        self.img_size = img_size
        # stage 1: full image -> (r_m, r_logv, cos_phi, sin_phi, phi_logv)
        self.recog_net_1 = MLP_Encoder(img_size * img_size, hidden, 5,
                                       nonlinearity=nonlinearity)
        # stage 2: derotated image -> (cos_theta, sin_theta, theta_logv)
        self.recog_net_2 = MLP_Encoder(img_size * img_size, hidden, 3,
                                       nonlinearity=nonlinearity)

    def forward(self, x):
        bs = x.shape[0]
        d = self.img_size
        device, dtype = x.device, x.dtype

        flat = x.reshape(bs, d * d)
        out1 = self.recog_net_1(flat)
        r_raw, r_logv, cos_phi_raw, sin_phi_raw, phi_logv = out1.split([1, 1, 1, 1, 1], dim=1)

        r_mean = torch.tanh(r_raw)              # normalized image coord in [-1, 1]
        r_var = torch.exp(r_logv) + 1e-4

        phi_vec_unnorm = torch.cat([cos_phi_raw, sin_phi_raw], dim=1)
        phi_vec = phi_vec_unnorm / (phi_vec_unnorm.norm(dim=-1, keepdim=True) + 1e-8)
        phi_conc = F.softplus(phi_logv) + 1.0

        cos_phi = phi_vec[:, 0]
        sin_phi = phi_vec[:, 1]
        r_scalar = r_mean[:, 0]

        # Warp so the CoM lands at (0,0) and the local-radial direction aligns with +x.
        # affine: output(x_out, y_out) queries input at R(phi) (x_out, y_out) + (r cos phi, r sin phi)
        theta_warp = get_theta(
            cos_phi, sin_phi,
            r_scalar * cos_phi, r_scalar * sin_phi,
            bs=bs, device=device, dtype=dtype,
        )
        grid = F.affine_grid(theta_warp, torch.Size((bs, 1, d, d)), align_corners=False)
        x_warped = F.grid_sample(x, grid, align_corners=False)

        out2 = self.recog_net_2(x_warped.reshape(bs, d * d))
        cos_th_raw, sin_th_raw, theta_logv = out2.split([1, 1, 1], dim=1)
        theta_vec_unnorm = torch.cat([cos_th_raw, sin_th_raw], dim=1)
        theta_vec = theta_vec_unnorm / (theta_vec_unnorm.norm(dim=-1, keepdim=True) + 1e-8)
        theta_conc = F.softplus(theta_logv) + 1.0

        return {
            'r_mean': r_mean, 'r_var': r_var,
            'phi_vec': phi_vec, 'phi_vec_raw': phi_vec_unnorm,
            'phi_concentration': phi_conc,
            'theta_vec': theta_vec, 'theta_vec_raw': theta_vec_unnorm,
            'theta_concentration': theta_conc,
            'x_warped': x_warped,   # exposed for debugging / visualization
        }


class DumbbellDecoder(nn.Module):
    """
    Input:  (r, cos_phi, sin_phi, cos_theta, sin_theta), each (bs, 1) or (bs,)
    Output: (bs, 1, 32, 32) reconstruction.

    The body sprite is a learned canonical dumbbell (head + tail along +x),
    warped by affine( R(phi + theta), translate (r cos phi, r sin phi) ).
    The planet sprite is a fixed (non-trainable) single bright pixel at the
    image center — there is nothing to learn about it, and making it learnable
    creates a degenerate equilibrium where the planet branch absorbs all the
    reconstruction signal and the body branch dies.
    """
    def __init__(self, img_size=32, hidden=100, nonlinearity='elu',
                 planet_brightness=0.9):
        super().__init__()
        self.img_size = img_size
        self.obs_net_body = MLP_Decoder(1, hidden, img_size * img_size,
                                        nonlinearity=nonlinearity)
        # Fixed planet sprite: a single bright pixel at the image center.
        planet_template = torch.zeros(1, 1, img_size, img_size)
        planet_template[0, 0, img_size // 2, img_size // 2] = planet_brightness
        self.register_buffer('planet_template', planet_template)

    def forward(self, r, cos_phi, sin_phi, cos_theta, sin_theta):
        # Accept either (bs,) or (bs, 1)
        r = r.view(-1)
        cos_phi = cos_phi.view(-1)
        sin_phi = sin_phi.view(-1)
        cos_theta = cos_theta.view(-1)
        sin_theta = sin_theta.view(-1)

        bs = r.shape[0]
        d = self.img_size
        device, dtype = r.device, r.dtype

        # Sanitize inputs — when called inside an ODE rollout, r and the (cos,
        # sin) pairs are unconstrained and can blow up. F.grid_sample on macOS
        # segfaults on NaN/Inf coordinates (instead of returning NaN), and even
        # finite-but-huge translations can break the autograd path.
        r = torch.nan_to_num(r, nan=0.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)
        cos_phi = torch.nan_to_num(cos_phi, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
        sin_phi = torch.nan_to_num(sin_phi, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
        cos_theta = torch.nan_to_num(cos_theta, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
        sin_theta = torch.nan_to_num(sin_theta, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
        norm_phi = torch.sqrt(cos_phi * cos_phi + sin_phi * sin_phi) + 1e-6
        cos_phi = cos_phi / norm_phi
        sin_phi = sin_phi / norm_phi
        norm_theta = torch.sqrt(cos_theta * cos_theta + sin_theta * sin_theta) + 1e-6
        cos_theta = cos_theta / norm_theta
        sin_theta = sin_theta / norm_theta

        ones = torch.ones(bs, 1, device=device, dtype=dtype)
        body_canon = self.obs_net_body(ones).view(bs, 1, d, d)

        # body angle in world frame = phi + theta
        cos_body = cos_phi * cos_theta - sin_phi * sin_theta
        sin_body = sin_phi * cos_theta + cos_phi * sin_theta

        # body is translated to (r cos phi, r sin phi) and rotated by body_angle
        theta_body = get_theta_inv(
            cos_body, sin_body,
            r * cos_phi, r * sin_phi,
            bs=bs, device=device, dtype=dtype,
        )
        grid_body = F.affine_grid(theta_body, torch.Size((bs, 1, d, d)), align_corners=False)
        body_warp = F.grid_sample(body_canon, grid_body, align_corners=False)

        # Planet is static at image center — broadcast the fixed template.
        planet = self.planet_template.expand(bs, 1, d, d)

        return body_warp + planet

    def init_body_canon(self, num_steps=200, lr=1e-3, verbose=True):
        """Pre-train obs_net_body so its constant output matches a synthetic
        dumbbell sprite. Call once before trainer.fit. See make_synthetic_body_canon."""
        device = self.planet_template.device
        target_np = make_synthetic_body_canon(img_size=self.img_size)
        target = torch.from_numpy(target_np).view(1, 1, self.img_size, self.img_size).to(device)

        opt = torch.optim.Adam(self.obs_net_body.parameters(), lr=lr)
        ones = torch.ones(1, 1, device=device)

        with torch.no_grad():
            initial_loss = F.mse_loss(
                self.obs_net_body(ones).view(1, 1, self.img_size, self.img_size), target
            ).item()
        for _ in range(num_steps):
            out = self.obs_net_body(ones).view(1, 1, self.img_size, self.img_size)
            loss = F.mse_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        final_loss = loss.item()
        if verbose:
            print(f'[init_body_canon] {num_steps} steps: MSE {initial_loss:.4f} -> {final_loss:.6f}')
        return final_loss
