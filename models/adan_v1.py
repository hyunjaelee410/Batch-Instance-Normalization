from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F

import torch
import torch.nn as nn


class ADAN_V1(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, batch=-1):
        super().__init__(num_features, eps, momentum, affine) # Set affine to false because weight size will be different
        self.batch = batch

        self.mean_cfc = Parameter(torch.Tensor(num_features, 3))
        self.mean_cfc.data.fill_(1)

        self.std_cfc = Parameter(torch.Tensor(num_features, 3))
        self.std_cfc.data.fill_(1)

        self.activation = nn.Sigmoid()
        setattr(self.mean_cfc, 'affine_gate', True)
        setattr(self.std_cfc, 'affine_gate', True)

    def get_mean_feat(self, feat):
        B, C, H, W = feat.size()

        feat = feat.view(B, C, -1)
        mean_i = feat.mean(dim=2, keepdim=True)

        if self.training:
            mean_b = mean_i.mean(dim=0, keepdim=True)

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_b.squeeze().data)

            mean_b = mean_b.repeat(B, 1, 1)
        else:
            mean_b = torch.autograd.Variable(self.running_mean).view(1, C, 1).repeat(B, 1, 1)

        feat_ones = torch.ones(B, C, 1).cuda()
        feat_zeros = torch.zeros(B, C, 1).cuda()

        mean_feat = torch.cat((feat_zeros, mean_i, feat_zeros), dim=2)
        #mean_feat = torch.cat((mean_i, mean_b, feat_ones), dim=2)

        return mean_feat

    def get_std_feat(self, feat):
        B, C, H, W = feat.size()

        feat = feat.view(B, C, -1)
        mean_i = feat.mean(dim=2, keepdim=True)
        var_i = feat.var(dim=2, keepdim=True)
        std_i = var_i.sqrt()

        mean_b = mean_i.mean(dim=0, keepdim=True)

        if self.training:
            temp = var_i + mean_i ** 2
            var_b = temp.mean(0, keepdim=True) - mean_b ** 2 + self.eps

            self.running_var.mul_(self.momentum)
            self.running_var.add_((1 - self.momentum) * var_b.squeeze().data)

            std_b = var_b.sqrt()
        else:
            var_b = torch.autograd.Variable(self.running_var).view(1, C, 1).repeat(B, 1, 1)
            std_b = var_b.sqrt()

        feat_ones = torch.ones(B, C, 1).cuda()
        feat_zeros = torch.zeros(B, C, 1).cuda()

        mean_feat = torch.cat((feat_zeros, std_i, feat_zeros), dim=2)
        #mean_feat = torch.cat((std_i, std_b, feat_ones), dim=2)
        return mean_feat

    def _forward(self, input):
        B, C, H, W = input.size()
        mean_feat = self.get_mean_feat(input)
        fake_mean = mean_feat * self.mean_cfc[None, :, :]
        fake_mean = torch.sum(fake_mean, dim=2)[:, :, None, None]

        std_feat = self.get_std_feat(input)
        fake_std = std_feat * self.std_cfc[None, :, :]
        fake_std = torch.sum(fake_std, dim=2)[:, :, None, None]

        input = (input - fake_mean) / (fake_std + self.eps)
        input.mul_(self.weight[None, :, None, None])
        input.add_(self.bias[None, :, None, None])

        return input

    def forward(self, input):
        B, C, H, W = input.size()

        # Compute BN (B x C x None)
        if self.batch == -1 or self.batch == B:
            out = self._forward(input)
        else:
            out = input.new(input.size())
            for i in range(0, B, self.batch):
                j = min(B, i + self.batch)
                out[i:j, :, :, :] = self.forward(input[i:j, :, :, :])
        return out

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}, ' \
                'batch={batch}'.format(**self.__dict__)
