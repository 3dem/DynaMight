import torch
import torch.nn


class PoseModule(torch.nn.Module):
    def __init__(self, box_size, device, orientations, translations):
        super(PoseModule, self).__init__()
        self.orientations = torch.nn.Parameter(orientations, requires_grad=True)
        self.translations = torch.nn.Parameter(translations, requires_grad=True)
        self.device = device
        self.box_size = box_size

    def forward(self, ind):
        r = self.orientations[ind]
        t = self.translations[ind]
        return r, t
