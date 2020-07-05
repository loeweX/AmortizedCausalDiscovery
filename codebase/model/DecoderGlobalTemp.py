from model.modules import *
from model.SimulationDecoder import SimulationDecoder
from model import utils


class MLPDecoderGlobalTemp(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self,
        n_in_node,
        edge_types,
        msg_hid,
        msg_out,
        n_hid,
        do_prob=0.0,
        skip_first=False,
        latent_dim=32,
    ):
        super(MLPDecoderGlobalTemp, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            # [nn.Linear(2 * n_in_node + latent_dim, msg_hid) for _ in range(edge_types)]
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)]
        )
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print("Using learned interaction net decoder.")

        self.dropout_prob = do_prob

    def single_step_forward(
        self,
        single_timestep_inputs,
        latents,
        rel_rec,
        rel_send,
        single_timestep_rel_type,
    ):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), pre_msg.size(2), self.msg_out_shape
        )

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i : i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, latents, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [
            rel_type.size(0),
            inputs.size(1),
            rel_type.size(1),
            rel_type.size(2),
        ]  # batch, sequence length, interactions between particles, interaction types
        rel_type = rel_type.unsqueeze(1).expand(
            sizes
        )  # copy relations over sequence length

        time_steps = inputs.size(1)
        assert pred_steps <= time_steps
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(
                last_pred, latents, rel_rec, rel_send, curr_rel_type
            )
            preds.append(last_pred)

        sizes = [
            preds[0].size(0),
            preds[0].size(1) * pred_steps,
            preds[0].size(2),
            preds[0].size(3),
        ]

        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, : (inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class SimulationDecoderGlobalTemp(SimulationDecoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
        super(SimulationDecoderGlobalTemp, self).__init__(
            loc_max, loc_min, vel_max, vel_min, suffix
        )

    def forward(self, inputs, relations, latents, rel_rec, rel_send, pred_steps=1):
        temperature = latents.unsqueeze(2)
        # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
        # Relation mx shape: [num_sims, num_things*num_things]

        # Only keep single dimension of softmax output
        relations = relations[:, :, 1]

        loc = inputs[:, :, :-1, :2].contiguous()
        vel = inputs[:, :, :-1, 2:].contiguous()

        # Broadcasting/shape tricks for parallel processing of time steps
        loc = loc.permute(0, 2, 1, 3).contiguous()
        vel = vel.permute(0, 2, 1, 3).contiguous()
        loc = loc.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)

        loc, vel = self.unnormalize(loc, vel)

        offdiag_indices = utils.get_offdiag_indices(inputs.size(1))
        edges = torch.zeros(relations.size(0), inputs.size(1) * inputs.size(1))

        if inputs.is_cuda:
            edges = edges.cuda()
            offdiag_indices = offdiag_indices.cuda()

        edges[:, offdiag_indices] = relations.float()

        edges = edges.view(relations.size(0), inputs.size(1), inputs.size(1))

        self.out = []

        for _ in range(0, self.sample_freq):
            x = loc[:, :, 0].unsqueeze(-1)
            y = loc[:, :, 1].unsqueeze(-1)

            xx = x.expand(x.size(0), x.size(1), x.size(1))
            yy = y.expand(y.size(0), y.size(1), y.size(1))
            dist_x = xx - xx.transpose(1, 2)
            dist_y = yy - yy.transpose(1, 2)

            if "_springs" in self.interaction_type:
                forces_size = -temperature * edges
                pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), -1)

                # Tricks for parallel processing of time steps
                pair_dist = pair_dist.view(
                    inputs.size(0),
                    (inputs.size(2) - 1),
                    inputs.size(1),
                    inputs.size(1),
                    2,
                )
                forces = (forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(3)
            else:  # charged particle sim
                e = (-1) * (edges * 2 - 1)
                forces_size = -temperature * e

                l2_dist_power3 = torch.pow(self.pairwise_sq_dist(loc), 3.0 / 2.0)
                l2_dist_power3 = self.set_diag_to_one(l2_dist_power3)

                l2_dist_power3 = l2_dist_power3.view(
                    inputs.size(0), (inputs.size(2) - 1), inputs.size(1), inputs.size(1)
                )
                forces_size = forces_size.unsqueeze(1) / (l2_dist_power3 + _EPS)

                pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), -1)
                pair_dist = pair_dist.view(
                    inputs.size(0),
                    (inputs.size(2) - 1),
                    inputs.size(1),
                    inputs.size(1),
                    2,
                )
                forces = (forces_size.unsqueeze(-1) * pair_dist).sum(3)

            forces = forces.view(
                inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2
            )

            if "_charged" in self.interaction_type:  # charged particle sim
                # Clip forces
                forces[forces > self._max_F] = self._max_F
                forces[forces < -self._max_F] = -self._max_F

            # Leapfrog integration step
            vel = vel + self._delta_T * forces
            loc = loc + self._delta_T * vel

            # Handle box boundaries
            loc, vel = self.clamp(loc, vel)

        loc, vel = self.renormalize(loc, vel)

        loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

        loc = loc.permute(0, 2, 1, 3)
        vel = vel.permute(0, 2, 1, 3)

        out = torch.cat((loc, vel), dim=-1)
        # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

        return out
