from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch


class BIN(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, batch=-1):
        super().__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)
        self.batch = batch

    def forward(self, input):
        B, C, H, W = input.size()
        
        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        
        # Compute BN (B x C x None)
        if self.batch == -1 or self.batch == B:
            out_bn = F.batch_norm(
                input, self.running_mean, self.running_var, bn_w, self.bias,
                self.training, self.momentum, self.eps)
        else:
            out_bn = input.new(input.size())
            for i in range(0, B, self.batch):
                j = min(B, i + self.batch)
                out_bn[i:j, :, :, :] = F.batch_norm(
                    input[i:j, :, :, :], self.running_mean, self.running_var, bn_w, self.bias,
                    self.training, self.momentum, self.eps)
        
        # Instance norm
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, B * C, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(B, C, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in
    
    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}, ' \
                'batch={batch}'.format(**self.__dict__)
