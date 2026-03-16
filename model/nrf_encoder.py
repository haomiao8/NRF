import torch
import torch.nn as nn
import torch.nn.functional as F


class NRF_Encoder(nn.Module):
    def __init__(
        self,
        latent_size,
        pe_dim,
        num_layers,
        hidden_layer_dim,
        latent_in, 
        dropout_layer,
        dropout_prob,
        weight_norm,
        weight_norm_layer
    ):
        super(NRF_Encoder, self).__init__()

        dims = [hidden_layer_dim] * (num_layers - 1)
        dims = [latent_size + pe_dim] + dims + [1]

        self.num_layers = num_layers
        self.norm_layers = set(weight_norm_layer or [])
        self.latent_in = set(latent_in or [])
        self.weight_norm = weight_norm
        self.pe_dim = pe_dim

        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            out_dim = dims[layer_idx + 1] - dims[0] if (layer_idx + 1) in self.latent_in else dims[layer_idx + 1]
            linear = nn.Linear(dims[layer_idx], out_dim)
            if self.weight_norm and layer_idx in self.norm_layers:
                linear = nn.utils.parametrizations.weight_norm(linear)
            self.layers.append(linear)

        self.relu = nn.ReLU()
        if dropout_prob:
            dropout_prob = float(dropout_prob)
        self.dropout_prob = dropout_prob
        self.dropout = set(dropout_layer or [])
    
    # input: N x (L+self.pe_dim)
    def forward(self, input):
        
        x = input

        for layer_idx, linear in enumerate(self.layers):
            if layer_idx in self.latent_in:
                x = torch.cat([x, input], 1)
            x = linear(x)
            if layer_idx < self.num_layers - 1:
                x = self.relu(x)

                if self.dropout and layer_idx in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        return x
    

    