#!/usr/bin/env python
"""
Delta-V training: learn perturbation potential with frozen base model.

Loads a pre-trained unperturbed model, HARD FREEZES everything
(encoder, decoder, V_net, M_net, g_net), and trains ONLY a small
delta_V_net on perturbed data.

V_total(cos_q, sin_q) = V_net_frozen(cos_q, sin_q) + delta_V_net(cos_q, sin_q)

Usage:
    python examples/pend_delta_v_frozen_trainer.py \
        --pretrained_ckpt results/pend/pend-lag-cavae-T_p=4-epoch=983-step=7871.ckpt \
        --data_path datasets/pendulum-gym-image-dataset-train-reverse-angle-perturbed.pkl \
        --name pend-delta-v-frozen \
        --max_epochs 1000 --annealing
"""
from argparse import ArgumentParser
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint

from lag_caVAE.lag_delta_v import Lag_Net_DeltaV
from lag_caVAE.nn_models import MLP_Encoder, MLP, PSD
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import my_collate, ImageDataset, HomoImageDataset

seed_everything(0)


class DeltaVFrozenModel(pl.LightningModule):
    """
    Perturbation identification with additive delta_V_net.

    ALL existing parameters are FROZEN (requires_grad=False).
    Only delta_V_net is trainable.
    """

    def __init__(self, hparams, data_path=None, pretrained_ckpt=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_path = data_path
        self.T_pred = self.hparams.T_pred
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        # --- Build full architecture ---
        self.recog_q_net = MLP_Encoder(32*32, 300, 3, nonlinearity='elu')
        self.obs_net = MLP_Encoder(1, 100, 32*32, nonlinearity='elu')

        V_net = MLP(2, 50, 1)
        g_net = MLP(2, 50, 1)
        M_net = PSD(2, 50, 1)

        # delta_V_net: same input space as V_net (cos_q, sin_q) → scalar
        delta_V_net = MLP(2, self.hparams.delta_v_hidden, 1, nonlinearity='tanh')

        # Initialize output layer near zero so delta_V starts as "no perturbation"
        # and gradually learns only the needed correction (residual learning)
        with torch.no_grad():
            delta_V_net.linear3.weight.mul_(0.01)
            delta_V_net.linear3.bias.mul_(0.01)

        self.ode = Lag_Net_DeltaV(
            q_dim=1, u_dim=1,
            g_net=g_net, M_net=M_net, V_net=V_net,
            delta_V_net=delta_V_net,
        )

        self.train_dataset = None
        self.non_ctrl_ind = 1

        if pretrained_ckpt is not None:
            self._load_and_freeze(pretrained_ckpt)

    def _load_and_freeze(self, ckpt_path):
        """Load pre-trained weights into base model, then freeze everything except delta_V_net."""
        from examples.pend_lag_cavae_trainer import Model as OriginalModel

        print(f"Loading pre-trained model from: {ckpt_path}")
        pretrained = OriginalModel.load_from_checkpoint(ckpt_path, map_location='cpu')

        # Copy weights for base model components
        self.recog_q_net.load_state_dict(pretrained.recog_q_net.state_dict())
        self.obs_net.load_state_dict(pretrained.obs_net.state_dict())
        self.ode.V_net.load_state_dict(pretrained.ode.V_net.state_dict())
        self.ode.M_net.load_state_dict(pretrained.ode.M_net.state_dict())
        self.ode.g_net.load_state_dict(pretrained.ode.g_net.state_dict())

        # HARD FREEZE everything except delta_V_net
        for name, param in self.named_parameters():
            if 'delta_V_net' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        delta_v_params = sum(p.numel() for p in self.ode.delta_V_net.parameters())
        print(f"  Trainable (delta_V_net): {trainable}")
        print(f"  Frozen (everything else): {frozen}")
        print(f"  delta_V_net parameters: {delta_v_params}")

    def train_dataloader(self):
        if self.hparams.homo_u:
            if self.train_dataset is None:
                self.train_dataset = HomoImageDataset(self.data_path, self.hparams.T_pred)
            if self.current_epoch < 1000:
                if self.current_epoch % 2 == 0:
                    u_idx = 0
                else:
                    u_idx = self.non_ctrl_ind
                    self.non_ctrl_ind += 1
                    if self.non_ctrl_ind == 9:
                        self.non_ctrl_ind = 1
            else:
                u_idx = self.current_epoch % 9
            self.train_dataset.u_idx = u_idx
            self.t_eval = torch.from_numpy(self.train_dataset.t_eval)
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                              shuffle=True, collate_fn=my_collate)
        else:
            train_dataset = ImageDataset(self.data_path, self.hparams.T_pred)
            self.t_eval = torch.from_numpy(train_dataset.t_eval)
            return DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                              shuffle=True, collate_fn=my_collate)

    def angle_vel_est(self, q0_m_n, q1_m_n, delta_t):
        delta_cos = q1_m_n[:, 0:1] - q0_m_n[:, 0:1]
        delta_sin = q1_m_n[:, 1:2] - q0_m_n[:, 1:2]
        q_dot0 = (-delta_cos * q0_m_n[:, 1:2] / delta_t
                  + delta_sin * q0_m_n[:, 0:1] / delta_t)
        return q_dot0

    def encode(self, batch_image):
        q_m_logv = self.recog_q_net(batch_image)
        q_m, q_logv = q_m_logv.split([2, 1], dim=1)
        q_m_n = q_m / q_m.norm(dim=-1, keepdim=True)
        q_v = F.softplus(q_logv) + 1
        return q_m, q_v, q_m_n

    def get_theta_inv(self, cos, sin, x, y, bs=None):
        bs = self.bs if bs is None else bs
        theta = torch.zeros([bs, 2, 3], dtype=self.dtype, device=self.device)
        theta[:, 0, 0] += cos
        theta[:, 0, 1] += -sin
        theta[:, 0, 2] += -x * cos + y * sin
        theta[:, 1, 0] += sin
        theta[:, 1, 1] += cos
        theta[:, 1, 2] += -x * sin - y * cos
        return theta

    def forward(self, X, u):
        [_, self.bs, d, d] = X.shape
        T = len(self.t_eval)

        # Encode
        self.q0_m, self.q0_v, self.q0_m_n = self.encode(X[0].reshape(self.bs, d*d))
        self.q1_m, self.q1_v, self.q1_m_n = self.encode(X[1].reshape(self.bs, d*d))

        # Reparameterize
        self.Q_q = VonMisesFisher(self.q0_m_n, self.q0_v)
        self.P_q = HypersphericalUniform(1, device=self.device)
        self.q0 = self.Q_q.rsample()
        while torch.isnan(self.q0).any():
            self.q0 = self.Q_q.rsample()

        # Estimate velocity
        self.q_dot0 = self.angle_vel_est(
            self.q0_m_n, self.q1_m_n, self.t_eval[1] - self.t_eval[0]
        )

        # ODE integration (uses Lag_Net_DeltaV with V_total = V + delta_V)
        z0_u = torch.cat((self.q0, self.q_dot0, u), dim=1)
        zT_u = odeint(self.ode, z0_u, self.t_eval, method=self.hparams.solver)
        self.qT, self.q_dotT, _ = zT_u.split([2, 1, 1], dim=-1)
        self.qT = self.qT.view(T * self.bs, 2)

        # Decode (coordinate-aware rendering)
        ones = torch.ones_like(self.qT[:, 0:1])
        self.content = self.obs_net(ones)
        theta = self.get_theta_inv(
            self.qT[:, 0], self.qT[:, 1], 0, 0, bs=T * self.bs
        )
        grid = F.affine_grid(theta, torch.Size((T * self.bs, 1, d, d)))
        self.Xrec = F.grid_sample(
            self.content.view(T * self.bs, 1, d, d), grid
        )
        self.Xrec = self.Xrec.view([T, self.bs, d, d])
        return None

    def training_step(self, train_batch, batch_idx):
        X, u = train_batch
        self.forward(X, u)

        # Image reconstruction loss (same as original model)
        lhood = -self.loss_fn(self.Xrec, X)
        lhood = lhood.sum([0, 2, 3]).mean()
        kl_q = torch.distributions.kl.kl_divergence(self.Q_q, self.P_q).mean()
        norm_penalty = (self.q0_m.norm(dim=-1).mean() - 1) ** 2

        # Zero-mean regularization: penalize DC offset in delta_V_net
        # The potential is defined up to a constant, so force zero mean
        if hasattr(self.ode, 'delta_V_q') and self.ode.delta_V_q is not None:
            zm_reg = self.ode.delta_V_q.mean() ** 2
        else:
            zm_reg = torch.tensor(0.0)

        lambda_ = self.current_epoch / 8000 if self.hparams.annealing else 1 / 100
        zm_weight = getattr(self.hparams, 'zm_lambda', 10.0)
        loss = -lhood + kl_q + lambda_ * norm_penalty + zm_weight * zm_reg

        logs = {
            'recon_loss': -lhood,
            'kl_q_loss': kl_q,
            'zm_reg': zm_reg,
            'train_loss': loss,
            'monitor': -lhood + kl_q,
        }
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # Only optimize delta_V_net parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=512, type=int)
        parser.add_argument('--delta_v_hidden', default=50, type=int,
                            help='Hidden dim for delta_V_net (default: 50, same as V_net)')
        parser.add_argument('--zm_lambda', default=10.0, type=float,
                            help='Zero-mean regularization weight for delta_V_net')
        return parser


def main(args):
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(
            PARENT_DIR, 'datasets',
            'pendulum-gym-image-dataset-train-reverse-angle-perturbed.pkl'
        )

    model = DeltaVFrozenModel(
        hparams=args,
        data_path=data_path,
        pretrained_ckpt=args.pretrained_ckpt,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='monitor',
        filepath=os.path.join(PARENT_DIR, 'logs', args.name,
                              args.name + f'-T_p={args.T_pred}'),
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer.from_argparse_args(
        args,
        deterministic=False,
        default_root_dir=os.path.join(PARENT_DIR, 'logs', args.name),
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--name', default='pend-delta-v-frozen', type=str)
    parser.add_argument('--T_pred', default=4, type=int)
    parser.add_argument('--solver', default='euler', type=str)
    parser.add_argument('--homo_u', dest='homo_u', action='store_true')
    parser.add_argument('--annealing', dest='annealing', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, required=True,
                        help='Path to unperturbed model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to perturbed training data')
    parser.set_defaults(homo_u=False, annealing=False)
    parser = Trainer.add_argparse_args(parser)
    parser = DeltaVFrozenModel.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
