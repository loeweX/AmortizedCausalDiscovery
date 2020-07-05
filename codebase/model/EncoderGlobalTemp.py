from model.modules import *
from model.MLPEncoder import MLPEncoder
from model.CNNEncoder import CNNEncoder


class CNNEncoderGlobalTemp(CNNEncoder):
    def __init__(
        self,
        args,
        n_in,
        n_hid,
        n_out,
        do_prob=0.0,
        factor=True,
        latent_dim=2,
        latent_sample_dim=1,
        num_atoms=5,
        num_timesteps=49,
    ):
        super().__init__(
            args,
            n_in,
            n_hid,
            n_out,
            do_prob,
            factor,
            n_in_mlp1=n_hid + latent_sample_dim,
        )

        self.mlp4_confounder = MLP(
            n_in * num_timesteps * num_atoms,
            n_hid,
            latent_dim,
            do_prob,
            use_batch_norm=False,
            final_linear=True,
        )
        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send):
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)

        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x_latent_input = inputs.view(inputs.size(0), 1, -1)
        latents = self.mlp4_confounder(x_latent_input).squeeze(1)

        inferred_mu, inferred_width = utils.get_uniform_parameters_from_latents(latents)
        latent_sample = utils.sample_uniform_from_latents(inferred_mu, inferred_width)
        l = latent_sample.view(latent_sample.size(0), 1, latent_sample.size(1)).repeat(
            1, x.size(1), 1
        )
        l = l.detach()
        # l = latents.view(latents.size(0), 1, latents.size(1)).repeat(1, x.size(1), 1)

        x = self.mlp1(torch.cat([x, l], 2))  # 2-layer ELU net per node
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x), latent_sample, inferred_mu, inferred_width
