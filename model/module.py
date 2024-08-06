import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Uniform


class LinearModule(nn.Module):
    """
    A combination of a linear layer followed with an activation function.
    """

    def __init__(self, data_dim, hidden_dim, activation_name, *args):
        super().__init__()

        assert isinstance(activation_name,
                          str), f"type of activation_name should be string while input is {type(activation_name)}"

        acti = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky relu": nn.LeakyReLU(args),
            "sin": Sine()
        }
        assert activation_name in acti.keys(), f"{[act for act in acti.keys()]} are used while input is {activation_name}"

        self.linear = nn.Linear(data_dim, hidden_dim)

        self.activation = acti[activation_name]

        # init weight and bias
        nn.init.normal_(self.linear.weight, std=0.1)

    def forward(self, x):
        y = self.linear(x)
        y = self.activation(y)
        return y


class ResModule(nn.Module):
    """
    Implementation of ResNet-featured module, a combination of which followed with a LinearModule.
    """

    def __init__(self, hidden_dim, num_layer, activation="tanh"):
        super().__init__()

        self.linear_layers = nn.ModuleList(
            [LinearModule(hidden_dim, hidden_dim, activation) for _ in range(num_layer - 1)])

        self.res = nn.Linear(hidden_dim, hidden_dim)

        self.output_linear_layer = LinearModule(hidden_dim, hidden_dim, activation)

    def forward(self, x):
        y = x
        for i, linear_layer in enumerate(self.linear_layers):
            y = linear_layer(y)
        y = y + self.res(x)
        y = self.output_linear_layer(y)
        return y


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        # 计算加权的平方误差
        loss = weight * (input - target) ** 2

        # 根据reduction参数选择返回的值
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
