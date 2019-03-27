from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F

import torch
import torch.nn as nn


class SAN_V1(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, batch=-1):
        super().__init__(num_features, eps, momentum, affine)
        self.batch = batch

        self.mean_cfc = Parameter(torch.Tensor(num_features, 3))
        self.mean_cfc.data.fill_(0)

        self.std_cfc = Parameter(torch.Tensor(num_features, 3))
        self.std_cfc.data.fill_(0)

        self.activation = nn.Sigmoid()
        setattr(self.mean_cfc, 'affine_gate', True)
        setattr(self.std_cfc, 'affine_gate', True)

    def get_feat(self, feat):
        B, C, H, W = feat.size()

        feat = feat.view(B, C, -1)
        mean_i = feat.mean(dim=2, keepdim=True)
        var_i = feat.var(dim=2, keepdim=True)
        std_i = var_i.sqrt()

        feat_ones = torch.ones(B, C, 1).cuda()

        mean_feat = torch.cat((mean_i, std_i, feat_ones), dim=2)

        return mean_feat

    def forward(self, input):
        B, C, H, W = input.size()

        feat = self.get_feat(input)
        mean_att = feat * self.mean_cfc[None, :, :]
        mean_att = torch.sum(mean_att, dim=2)[:, :, None, None]
        mean_att = self.activation(mean_att)

        std_att = feat * self.std_cfc[None, :, :]
        std_att = torch.sum(std_att, dim=2)[:, :, None, None]
        std_att = self.activation(std_att)

        # Compute BN (B x C x None)
        if self.batch == -1 or self.batch == B:
            out = F.batch_norm(
                    input, self.running_mean, self.running_var, None, None,
                    self.training, self.momentum, self.eps)
        else:
            out = input.new(input.size())
            for i in range(0, B, self.batch):
                j = min(B, i + self.batch)
                out[i:j, :, :, :] = F.batch_norm(
                    input[i:j, :, :, :], self.running_mean, self.running_var, None, None,
                    self.training, self.momentum, self.eps)

        out.mul_(self.weight[None, :, None, None] * std_att)
        out.add_(self.bias[None, :, None, None] * mean_att)
        out = out.view(B, C, H, W)
        
        return out

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}, ' \
                'batch={batch}'.format(**self.__dict__)
