import torch
import torch.nn as nn
import torch.nn.functional as F

from model import utils, utils_unobserved
from model.MLPEncoder import MLPEncoder

_EPS = 1e-10

class MLPEncoderUnobserved(MLPEncoder):
    def __init__(self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super().__init__(args, n_in, n_hid, n_out, do_prob, factor)

        self.unobserved = args.unobserved

        self.lstm1 = nn.LSTM(
            (args.num_atoms - self.unobserved) * args.dims,
            n_hid,
            bidirectional=True,
            dropout=do_prob,
        )
        self.lstm2 = nn.LSTM(n_hid * 2, args.dims, bidirectional=False, dropout=do_prob)

        self.init_weights()
        print("Using unobserved encoder.")

    def evaluate_unobserved(self, unobserved, target):
        return F.mse_loss(torch.squeeze(unobserved), torch.squeeze(target))

    def calc_unobserved_q(self, unobserved):
        ### Gaussian prior
        unobserved_mu = self.fc_mu(unobserved)
        unobserved_log_sigma = self.fc_logsigma(unobserved)

        unobserved = utils.sample_normal_from_latents(
            unobserved_mu,
            unobserved_log_sigma,
            downscale_factor=self.args.prior_downscale,
        )

        loss_kl_latent = utils.kl_normal_reverse(
            0,
            1,
            unobserved_mu,
            unobserved_log_sigma,
            downscale_factor=self.args.prior_downscale,
        )
        return unobserved, loss_kl_latent

    def forward(self, inputs, rel_rec, rel_send, mask_idx=0):
        timesteps = inputs.size(2)

        # input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        observed = utils_unobserved.remove_unobserved(self.args, inputs, mask_idx)

        observed = observed.permute(2, 0, 1, 3)
        observed = observed.reshape(observed.size(0), observed.size(1), -1)
        unobserved, _ = self.lstm1(observed)
        unobserved, _ = self.lstm2(unobserved)
        unobserved = unobserved.unsqueeze(0).permute(2, 0, 1, 3)
        unobserved = torch.reshape(
            unobserved, [unobserved.size(0), unobserved.size(1), timesteps, -1]
        )
        # output shape: [num_sims, num_atoms, num_timesteps, num_dims]

        target_unobserved = inputs[:, mask_idx, :, :]
        mse_unobserved = self.evaluate_unobserved(unobserved, target_unobserved)

        data_encoder = torch.cat(
            (inputs[:, :mask_idx, :], unobserved, inputs[:, mask_idx + 1 :, :],), dim=1,
        )

        output = super().forward(data_encoder, rel_rec, rel_send)

        return (output, unobserved, mse_unobserved)
