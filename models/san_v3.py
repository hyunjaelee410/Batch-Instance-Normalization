from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F

import torch
import torch.nn as nn


class SAN_V3(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, batch=-1):
        super().__init__(num_features, eps, momentum, affine)
        self.batch = batch

        z_dim = 8

        self.mean_cfc_1 = Parameter(torch.Tensor(num_features, 3, z_dim))
        self.mean_cfc_2 = Parameter(torch.Tensor(num_features, z_dim+1, 1))
        self.mean_cfc_1.data.fill_(0)
        self.mean_cfc_2.data.fill_(0)

        self.std_cfc_1 = Parameter(torch.Tensor(num_features, 3, z_dim))
        self.std_cfc_2 = Parameter(torch.Tensor(num_features, z_dim+1, 1))
        self.std_cfc_1.data.fill_(0)
        self.std_cfc_2.data.fill_(0)

        self.activation = nn.Sigmoid()
        setattr(self.mean_cfc_1, 'affine_gate', True)
        setattr(self.mean_cfc_2, 'affine_gate', True)
        setattr(self.std_cfc_1, 'affine_gate', True)
        setattr(self.std_cfc_2, 'affine_gate', True)

    def get_feat(self, feat):
        B, C, H, W = feat.size()

        feat = feat.view(B, C, -1)
        mean_i = feat.mean(dim=2, keepdim=True)
        var_i = feat.var(dim=2, keepdim=True)
        std_i = var_i.sqrt()

        feat_ones = torch.ones(B, C, 1).cuda()

        mean_feat = torch.cat((mean_i, std_i, feat_ones), dim=2)
        
        return torch.transpose(mean_feat, 0, 1)

    def forward(self, input):
        B, C, H, W = input.size()
        feat_ones = torch.ones(C, B, 1).cuda()

        feat = self.get_feat(input)

        mean_att_z = torch.bmm(feat, self.mean_cfc_1)
        mean_att_z = torch.cat((mean_att_z, feat_ones), dim=2)
        mean_att = torch.bmm(mean_att_z, self.mean_cfc_2)
        mean_att = torch.transpose(mean_att, 0, 1).squeeze()
        mean_att = self.activation(mean_att)

        std_att_z = torch.bmm(feat, self.std_cfc_1)
        std_att_z = torch.cat((std_att_z, feat_ones), dim=2)
        std_att = torch.bmm(std_att_z, self.std_cfc_2)
        std_att = torch.transpose(std_att, 0, 1).squeeze()
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

        out.mul_(self.weight[None, :, None, None] * std_att[:, :, None, None])
        out.add_(self.bias[None, :, None, None] * mean_att[:, :, None, None])
        out = out.view(B, C, H, W)
        
        return out

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}, ' \
                'batch={batch}'.format(**self.__dict__)
