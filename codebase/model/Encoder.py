from abc import abstractmethod
import torch

from model.modules import *


class Encoder(nn.Module):
    def __init__(self, args, factor=True):
        super(Encoder, self).__init__()
        self.args = args
        self.factor = factor

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(
            inputs.size(0) * receivers.size(1), inputs.size(2), inputs.size(3)
        )
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(
            inputs.size(0) * senders.size(1), inputs.size(2), inputs.size(3)
        )
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders,receivers], dim=2)
        return edges

    @abstractmethod
    def forward(self, inputs, rel_rec, rel_send, mask_idx=None):
        pass