import torch
import rff


class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return torch.sin(x)


### Copied from Amiranashvili
class RegistrationINR(torch.nn.Module):
    def __init__(self, num_layers, layers_with_coords, dropout_p,  activation, hidden_layers, encoding):
        def block(num_ch_in, num_ch_out):
            if activation == "sine":
                return torch.nn.Sequential(
                    torch.nn.Linear(num_ch_in, num_ch_out), Sin()
                )
            else:
                return torch.nn.Sequential(
                    torch.nn.Linear(num_ch_in, num_ch_out), torch.nn.ReLU(True)
                )
        super().__init__()
        self.encode = False
        spatial_dim = 3
        if encoding != None:
            self.encode = True
            frequencies = encoding["frequencies"]
            sigma = encoding["sigma"]
            self.encoding = rff.layers.PositionalEncoding(sigma=sigma, m=frequencies)
            spatial_dim = 6 * frequencies
        self.activation = activation
        self.layers_with_coords = layers_with_coords
        in_channels = [hidden_layers] * num_layers
        channels_with_coords = hidden_layers + spatial_dim
        for lyr_id in self.layers_with_coords:
            in_channels[lyr_id] = channels_with_coords
        in_channels[0] = spatial_dim
        self.res_layers = torch.nn.ModuleList(
            [block(in_channels[i], hidden_layers) for i in range(num_layers - 1)])
        self.last_layer = torch.nn.Linear(in_channels[-1], 3)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropout_layers = [int(num_layers / 4), int(3 * num_layers / 4)]

    def forward(self, local_coords):
        if self.encode:
            local_coords = self.encoding(local_coords)
        features = torch.Tensor().to(device='cuda')
        for i, layer in enumerate(self.res_layers):
            append_coords = i in self.layers_with_coords
            dropout = i in self.dropout_layers
            if append_coords:
                features = torch.cat([features, local_coords], dim=-1)
            if dropout:
                out = self.dropout(layer(features.to(dtype=torch.float32, device="cuda")))
            else:
                out = layer(features.to(dtype=torch.float32, device="cuda"))
            features = out if append_coords else features + out
        features = self.last_layer(features)
        return features


def create_model(params, wandb):
    dropout_p = wandb.config.dropout
    activation = wandb.config.activation_function
    hidden_layers = wandb.config.hidden_layer_size
    encoding = None
    if wandb.config.encoding:
        encoding = {
            'sigma': 0.5,
            'frequencies': 21
        }
    op_num_layers = params['op_num_layers']
    op_coord_layers = params['op_coord_layers']
    net = RegistrationINR(op_num_layers, op_coord_layers, dropout_p, activation, hidden_layers, encoding)
    return net
