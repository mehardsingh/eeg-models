import torch.nn as nn

from layers import Conv2dWithConstraint, LinearWithConstraint
from utils import initialize_weight


class EEGNet(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()

        n_classes = config.n_classes
        s = config.s
        F1 = config.F1
        D = config.D
        F2 = config.F2
        T1 = config.T1
        T2 = config.T2
        P1 = config.P1
        P2 = config.P2
        drop_out = config.drop_out
        pool_mode = config.pool_mode
        weight_init_method = config.weight_init_method
        last_dim = config.last_dim

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1), bias=False, padding='same'),
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (s, 1), bias=False, groups=F1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1)),
            nn.Dropout(drop_out)
        )

        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2), bias=False, padding='same', groups=F1 * D),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, P2)),
            nn.Dropout(drop_out)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(last_dim, n_classes, max_norm=0.25)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        
        # print(x.shape)
        out = self.spectral(x)
        # print(out.shape)
        out = self.spatial(out)
        # print(out.shape)
        out = self.temporal(out)
        # print(out.shape)
        out = self.classifier(out)
        return out