import timm
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = timm.create_model(args.encoder_name, pretrained=args.pretrained, drop_path_rate=args.drop_path_rate, num_classes=args.num_classes)

    def forward(self, x):
        return self.net(x)

# For Noisy Student
class StochasticDepth(nn.Module):
    def __init__(self, module: nn.Module, p: float = 0.5, training=True):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError(
                "Stochastic Depth p has to be between 0 and 1 but got {}".format(p)
            )
        self.module = module
        self.p = p
        self._sampler = torch.Tensor(1)
        self.training = training

    def forward(self, inputs):
        if self.training and self._sampler.uniform_():
            return inputs
        return self.p * self.module(inputs)