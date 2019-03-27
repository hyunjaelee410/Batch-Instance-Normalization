from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F

import torch
import torch.nn as nn


class BN(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, batch=-1):
        super().__init__(num_features, eps, momentum, affine)
        self.batch = batch

    def forward(self, input):
        B, C, H, W = input.size()

        # Compute BN (B x C x None)
        if self.batch == -1 or self.batch == B:
            out = F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training, self.momentum, self.eps)
        else:
            out = input.new(input.size())
            for i in range(0, B, self.batch):
                j = min(B, i + self.batch)
                out[i:j, :, :, :] = F.batch_norm(
                    input[i:j, :, :, :], self.running_mean, self.running_var, self.weight, self.bias,
                    self.training, self.momentum, self.eps)
        return out

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}, ' \
                'batch={batch}'.format(**self.__dict__)
