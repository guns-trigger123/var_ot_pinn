import torch.nn as nn


class PINN_FCN(nn.Module):
    def __init__(self, data_dim, output_dim):
        super().__init__()
        num_nerons = 64
        self.fcn = nn.Sequential(
            nn.Linear(data_dim, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, output_dim),
        )

    def forward(self, x):
        for layer in self.fcn:
            x = layer(x)
        return x
