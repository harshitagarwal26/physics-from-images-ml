"""
Stage-1 trainer for the rigid-dumbbell caVAE.

Learns:
    - DumbbellEncoder  (image -> r, phi, theta posteriors)
    - DumbbellDecoder  (r, phi, theta -> image)
    - Lag_Net_Dumbbell (M_net, V_net, g_net) integrated with torchdiffeq

Loss: negative ELBO
    - reconstruction  : MSE of decoded T-step rollout vs. observed frames
    - KL on r         : Normal(r_m, r_v) || N(0, 1)
    - KL on phi       : VonMisesFisher(phi_m_n, phi_conc) || HypersphericalUniform
    - KL on theta     : VonMisesFisher(theta_m_n, theta_conc) || HypersphericalUniform
    - raw-norm penalty (so the pre-normalization (cos, sin) vectors stay near unit length)
"""

from argparse import ArgumentParser
import os, sys

# macOS conda envs frequently load libomp twice (PyTorch + scipy/MKL),
# producing a segfault before training starts. Set these before importing
# torch to keep things stable.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Print a Python stack trace on the next segfault so we have something to
# diagnose if it recurs.
import faulthandler
faulthandler.enable()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
# Single-threaded torch — eliminates intra-op races inside the heavy
# create_graph=True autograd path of Lag_Net_Dumbbell.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint
from torch.distributions.normal import Normal

from lag_caVAE.vae_dumbbell import DumbbellEncoder, DumbbellDecoder
from lag_caVAE.lag_dumbbell import Lag_Net_Dumbbell
from lag_caVAE.nn_models import MLP, PSD, MatrixNet
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import ImageDataset, my_collate

seed_everything(0)


class Model(pl.LightningModule):

    def __init__(self, hparams, data_path=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_path = data_path
        self.T_pred = self.hparams.T_pred
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        self.encoder = DumbbellEncoder(img_size=32, hidden=300, nonlinearity='elu')
        self.decoder = DumbbellDecoder(img_size=32, hidden=100, nonlinearity='elu')

        V_net = MLP(3, 100, 1)
        M_net = PSD(3, 300, 3)
        g_net = MatrixNet(3, 100, 3, shape=(3, 1))
        self.ode = Lag_Net_Dumbbell(g_net=g_net, M_net=M_net, V_net=V_net, u_dim=1)

    def train_dataloader(self):
        train_dataset = ImageDataset(self.data_path, self.hparams.T_pred, ctrl=True)
        self.t_eval = torch.from_numpy(train_dataset.t_eval)
        return DataLoader(
            train_dataset, batch_size=self.hparams.batch_size,
            shuffle=True, collate_fn=my_collate,
        )

    @staticmethod
    def _angle_vel_from_vec(v0, v1, dt):
        """Finite-difference angular velocity from two unit (cos, sin) vectors."""
        dcos = v1[:, 0:1] - v0[:, 0:1]
        dsin = v1[:, 1:2] - v0[:, 1:2]
        # d(sin)/dt = cos*omega, d(cos)/dt = -sin*omega -> omega = (-dcos*sin + dsin*cos)/dt
        return (-dcos * v0[:, 1:2] + dsin * v0[:, 0:1]) / dt

    def _encode_batch(self, img):
        """img: (bs, 32, 32)  ->  full encoder output dict."""
        return self.encoder(img.unsqueeze(1))

    def forward(self, X, u):
        # X: (T_pred+1, bs, 32, 32)
        T_pred1, bs, d, _ = X.shape
        self.bs = bs
        T = len(self.t_eval)
        dt = self.t_eval[1] - self.t_eval[0]

        # Encode the first two frames — initial pose + finite-difference velocities.
        out0 = self._encode_batch(X[0])
        out1 = self._encode_batch(X[1])

        self.r0_m, self.r0_v = out0['r_mean'], out0['r_var']
        self.phi0_vec, self.phi0_raw, self.phi0_conc = (
            out0['phi_vec'], out0['phi_vec_raw'], out0['phi_concentration']
        )
        self.theta0_vec, self.theta0_raw, self.theta0_conc = (
            out0['theta_vec'], out0['theta_vec_raw'], out0['theta_concentration']
        )

        r1_m = out1['r_mean']
        phi1_vec = out1['phi_vec']
        theta1_vec = out1['theta_vec']

        # Reparameterize
        self.Q_r0 = Normal(self.r0_m, self.r0_v)
        self.P_normal = Normal(torch.zeros_like(self.r0_m), torch.ones_like(self.r0_v))
        r0 = self.Q_r0.rsample()

        self.P_hyper_uni = HypersphericalUniform(1, device=self.device)

        self.Q_phi0 = VonMisesFisher(self.phi0_vec, self.phi0_conc)
        phi0 = self.Q_phi0.rsample()
        while torch.isnan(phi0).any():
            phi0 = self.Q_phi0.rsample()

        self.Q_theta0 = VonMisesFisher(self.theta0_vec, self.theta0_conc)
        theta0 = self.Q_theta0.rsample()
        while torch.isnan(theta0).any():
            theta0 = self.Q_theta0.rsample()

        # Velocity estimates (from posterior means of frames 0, 1)
        r_dot0 = (r1_m - self.r0_m) / dt
        phi_dot0 = self._angle_vel_from_vec(self.phi0_vec, phi1_vec, dt)
        theta_dot0 = self._angle_vel_from_vec(self.theta0_vec, theta1_vec, dt)

        # Build ODE initial state and integrate
        z0_u = torch.cat([
            r0, phi0, theta0,
            r_dot0, phi_dot0, theta_dot0,
            u,
        ], dim=1)                                    # (bs, 9)
        zT_u = odeint(self.ode, z0_u, self.t_eval, method=self.hparams.solver)
        # zT_u: (T, bs, 9)

        # Split back out the pose coordinates.
        r_traj = zT_u[..., 0:1]              # (T, bs, 1)
        cos_phi_traj = zT_u[..., 1:2]
        sin_phi_traj = zT_u[..., 2:3]
        cos_theta_traj = zT_u[..., 3:4]
        sin_theta_traj = zT_u[..., 4:5]

        # Decode every timestep.
        r_flat = r_traj.reshape(T * bs, 1)
        cp_flat = cos_phi_traj.reshape(T * bs, 1)
        sp_flat = sin_phi_traj.reshape(T * bs, 1)
        ct_flat = cos_theta_traj.reshape(T * bs, 1)
        st_flat = sin_theta_traj.reshape(T * bs, 1)
        Xrec_flat = self.decoder(r_flat, cp_flat, sp_flat, ct_flat, st_flat)
        self.Xrec = Xrec_flat.view(T, bs, d, d)        # (T, bs, 32, 32)
        return None

    def training_step(self, train_batch, batch_idx):
        X, u = train_batch
        self.forward(X, u)

        lhood = -self.loss_fn(self.Xrec, X)               # (T, bs, d, d)
        lhood = lhood.sum([0, 2, 3]).mean()                # sum over T, pixels; mean over batch

        kl_r = torch.distributions.kl.kl_divergence(self.Q_r0, self.P_normal).mean()
        kl_phi = torch.distributions.kl.kl_divergence(self.Q_phi0, self.P_hyper_uni).mean()
        kl_theta = torch.distributions.kl.kl_divergence(self.Q_theta0, self.P_hyper_uni).mean()
        kl_q = kl_r + kl_phi + kl_theta

        norm_penalty = (
            (self.phi0_raw.norm(dim=-1).mean() - 1) ** 2
            + (self.theta0_raw.norm(dim=-1).mean() - 1) ** 2
        )

        # KL warm-up + cap. Beta is capped at 0.1: full beta=1 flattens the vMF
        # posterior on theta, which makes head/tail interchangeable and erodes
        # the asymmetric brightness in body_canon (tail peak fell to 0.25 vs
        # synthetic target 0.6). Stage-1 prioritizes reconstruction; the prior
        # matters more once the latent is used for dynamics.
        beta = min(0.1, self.current_epoch / 200)
        lambda_ = (self.current_epoch / 8000) if self.hparams.annealing else (1 / 100)
        loss = -lhood + beta * kl_q + lambda_ * norm_penalty

        logs = {
            'recon_loss': -lhood, 'kl_r': kl_r, 'kl_phi': kl_phi, 'kl_theta': kl_theta,
            'train_loss': loss, 'monitor': -lhood + kl_q,
        }
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=512, type=int)
        return parser


def main(args):
    model = Model(
        hparams=args,
        data_path=os.path.join(PARENT_DIR, 'datasets', args.data_name),
    )
    # Break the encoder/decoder bootstrap deadlock: seed body_canon with a
    # synthetic dumbbell sprite so the encoder has a meaningful target to
    # align against from epoch 0.
    model.decoder.init_body_canon()
    checkpoint_callback = ModelCheckpoint(
        monitor='monitor',
        filepath=os.path.join(
            PARENT_DIR, 'logs', args.name,
            args.name + f'-T_p={args.T_pred}',
        ),
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        default_root_dir=os.path.join(PARENT_DIR, 'logs', args.name),
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--name', default='dumbbell-lag-cavae', type=str)
    parser.add_argument('--T_pred', default=4, type=int)
    parser.add_argument('--solver', default='euler', type=str)
    parser.add_argument('--annealing', dest='annealing', action='store_true')
    parser.add_argument('--data_name', default='dumbbell-rigid-dataset.pkl', type=str)
    parser.set_defaults(annealing=False)
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
