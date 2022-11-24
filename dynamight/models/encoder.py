import torch.nn
from torch import nn as nn


class ResHetEncoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim):
        super().__init__()  # call of torch.nn.Module
        # self.down = nn.AvgPool2d(2,stride = 2)
        self.lin1 = nn.Linear(in_features=box_size ** 2, out_features=256)
        self.lin1a = nn.Linear(in_features=256, out_features=256)
        self.lin1b = nn.Linear(in_features=256, out_features=128)
        self.lin1c = nn.Linear(in_features=128, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=latent_dim)
        self.lin3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        # self.enc_optimizer = []

    def forward(self, x):
        # inp = self.down(x)
        inp = self.flat(x)
        x2 = self.lin1(inp)
        x2 = x2 + self.actr(self.lin1a(x2))
        x2 = self.actr(x2)
        x2 = self.act1(self.lin1b(x2))
        x2 = x2 + self.act1(self.lin1c(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        return x2, x3


class HetEncoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim, down_sample):
        super().__init__()  # call of torch.nn.Module
        self.down_sample = down_sample
        self.down = nn.AvgPool2d(2, stride=2)
        self.lin1 = nn.Linear(in_features=int((box_size / (2 ** down_sample)) ** 2),
                              out_features=256)
        # self.lin1a = nn.Linear(in_features = 256, out_features = 256)
        # self.lin1aa = nn.Linear(in_features = 256, out_features = 256)
        self.lin1b = nn.Linear(in_features=256, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=latent_dim)
        self.lin3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        self.latent_dim = latent_dim
        # self.enc_optimizer = []

    def forward(self, x, ctf):
        # inp = self.down(x)
        # inp = torch.fft.fft2(x,dim=[-1,-2],norm = 'ortho')
        # inp = x*ctf
        # inp = torch.real(torch.fft.ifft2(inp,dim=[-1,-2],norm = 'ortho')).float()
        inp = x
        for i in range(self.down_sample):
            inp = self.down(inp)
        inp = self.flat(inp)
        x2 = self.lin1(inp)
        x2 = self.actr(x2)
        # x2 = self.actr(self.lin1a(x2))
        # x2 = self.actr(self.lin1aa(x2))
        x2 = self.act1(self.lin1b(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        return x2, x3  #
