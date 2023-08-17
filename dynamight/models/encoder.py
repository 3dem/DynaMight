import torch.nn
from torch import nn as nn
import numpy as np


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


class N2F_Encoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim, down_sample):
        super().__init__()  # call of torch.nn.Module
        self.down_sample = down_sample
        self.down = nn.AvgPool2d(2, stride=2)
        self.lin1 = nn.Linear(in_features=int((box_size / (4 ** down_sample)) ** 2),
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
        self.N2F_block = N2F_block(box_size)
        # self.enc_optimizer = []

    def forward(self, x, ctf):
        # inp = self.down(x)
        # inp = torch.fft.fft2(x,dim=[-1,-2],norm = 'ortho')
        # inp = x*ctf
        # inp = torch.real(torch.fft.ifft2(inp,dim=[-1,-2],norm = 'ortho')).float()
        out1, lab = self.N2F_block(x)
        inp = x
        denois, labdenois = self.N2F_block(inp, train=False)
        inp = self.flat(denois)
        x2 = self.lin1(inp)
        x2 = self.actr(x2)
        # x2 = self.actr(self.lin1a(x2))
        # x2 = self.actr(self.lin1aa(x2))
        x2 = self.act1(self.lin1b(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        return x2, x3, denois, out1.squeeze(), lab  #


class N2F_block(torch.nn.Module):
    def __init__(self, box_size):
        super().__init__()
        self.box_size = box_size
        self.down = nn.AvgPool2d(4, stride=4)
        self.net = Net()

    def forward(self, x, train=True):
        inp = self.down(x)
        inp = inp-torch.min(inp)
        inp = inp/torch.max(inp)
        if train == True:
            choice = np.random.randint(0, 4)
            if choice == 0:
                inp1 = inp[:, ::2, :]
                lab = inp[:, 1::2, :]
            elif choice == 1:
                inp1 = inp[:, :, ::2]
                lab = inp[:, :, 1::2]
            elif choice == 2:
                inp1 = inp[:, 1::2, :]
                lab = inp[:, ::2, :]
            elif choice == 3:
                inp1 = inp[:, :, 1::2]
                lab = inp[:, :, ::2]
        else:
            inp1 = inp
            lab = inp
        out = self.net(inp1.unsqueeze(1))

        return out, lab


class TwoCon(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoCon(1, 64)
        self.conv2 = TwoCon(64, 64)
        self.conv3 = TwoCon(64, 64)
        self.conv4 = TwoCon(64, 64)
        self.conv6 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        x = torch.sigmoid(self.conv6(x))
        return x
