import torch
import torch.nn as nn

from layers import Conv2dWithConstraint, LinearWithConstraint
from utils import initialize_weight

torch.set_printoptions(linewidth=1000)


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            config
    ):
        super(ShallowConvNet, self).__init__()

        n_classes = config.n_classes
        s = config.s
        F1 = config.F1
        T1 = config.T1
        F2 = config.F2
        P1_T = config.P1_T
        P1_S = config.P1_S
        drop_out = config.drop_out
        pool_mode = config.pool_mode
        weight_init_method = config.weight_init_method
        last_dim = config.last_dim

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, T1), max_norm=2),
            Conv2dWithConstraint(F1, F2, (s, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, P1_T), (1, P1_S)),
            ActLog(),
            nn.Dropout(drop_out),
            nn.Flatten(),
            LinearWithConstraint(last_dim, n_classes, max_norm=0.5)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        
        out = self.net(x)
        return out


class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))