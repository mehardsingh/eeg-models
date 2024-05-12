from typing import List

import torch
import torch.nn as nn

from layers import Conv2dWithConstraint, LinearWithConstraint
from utils import initialize_weight


class DeepConvNet(nn.Module):
    def __init__(
            self,
            config
    ) -> None:
        super(DeepConvNet, self).__init__()

        n_classes = config.n_classes
        s = config.s
        first_conv_length = config.first_conv_length
        block_out_channels = config.block_out_channels
        pool_size = config.pool_size
        last_dim = config.last_dim
        weight_init_method = config.weight_init_method

        self.first_conv_block = nn.Sequential(
            Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, first_conv_length), max_norm=2),
            Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(s, 1), bias=False,
                                 max_norm=2),
            nn.BatchNorm2d(block_out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.deep_block = nn.ModuleList(
            [self.default_block(block_out_channels[i - 1], block_out_channels[i], first_conv_length, pool_size) for i in
             range(2, 5)]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(last_dim, n_classes, max_norm=0.5)  # time points = 1125
        )

        initialize_weight(self, weight_init_method)

    def default_block(self, in_channels, out_channels, T, P):
        default_block = nn.Sequential(
            nn.Dropout(0.5),
            Conv2dWithConstraint(in_channels, out_channels, (1, T), bias=False, max_norm=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d((1, P))
        )
        return default_block

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        # print(x.shape)
        out = self.first_conv_block(x)
        # print(out.shape)
        for block in self.deep_block:
            # print(out.shape)
            out = block(out)
        # print(out.shape)
        out = self.classifier(out)
        return out