# https://arxiv.org/pdf/2104.01233.pdf

# import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

from layers import Conv2dWithConstraint, LinearWithConstraint, Swish, LogVarLayer
from utils import initialize_weight

from julius.filters import bandpass_filter

# current_module = sys.modules[__name__]


class FBCNet(nn.Module):
    def __init__(self,
                 config
                 ):
        super().__init__()

        n_classes = config.n_classes
        n_electrode = config.n_electrode
        m = config.m
        temporal_stride = config.temporal_stride
        weight_init_method = config.weight_init_method
        self.band_freqs = config.band_freqs
        self.fs = config.fs
        n_band = len(self.band_freqs)


        self.temporal_stride = temporal_stride

        # SCB (Spatial Convolution Block)
        self.scb = nn.Sequential(
            Conv2dWithConstraint(n_band, m * n_band, (n_electrode, 1), groups=n_band, max_norm=2),
            nn.BatchNorm2d(m * n_band),
            Swish()
        )

        # Temporal Layer
        self.temporal_layer = LogVarLayer(-1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(n_band * m * temporal_stride, n_classes, max_norm=0.5)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        # batch_size, time_points, n_electrode
        x = x.permute(0, 2, 1) # batch_size, n_electrode, time_points
        shape = x.shape

        x = x.reshape(-1, x.size(-1)) # (batch_size x n_electrode), time_points

        bands = list()
        for i in range(len(self.band_freqs)):
            band_thresh = self.band_freqs[i]
            bands.append(bandpass_filter(x, band_thresh[0]/self.fs, band_thresh[1]/self.fs).reshape(shape))
        x = torch.stack(bands, dim=0) # n_bands x batch_size x n_electrode x time_points
        x = x.permute(1, 0, 2, 3)

        print(x.shape)

        # batch_size, n_band, n_electrode, time_points
        out = self.scb(x)
        out = F.pad(out, (0, 3))
        out = out.reshape([*out.shape[:2], self.temporal_stride, int(out.shape[-1] / self.temporal_stride)])
        out = self.temporal_layer(out)
        out = self.classifier(out)
        return out