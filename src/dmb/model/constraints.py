import torch


class Exponential(torch.nn.Module):

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.exp(x) + self.eps
