#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:43:55 2021

@author: schwab
"""

import torch
import torch.nn as nn
import torch.fft


class ResBlock(torch.nn.Module):
    """Multiple residual layers."""
    def __init__(
        self,
        pos_enc_dim: int,
        latent_dim: int,
        n_neurons: int,
        n_layers: int,
        box_size: int
    ):
        super(ResBlock, self).__init__()
        self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        self.lin1 = nn.Linear(n_neurons, n_neurons, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        self.act = nn.ELU()
        self.n_layers = n_layers
        self.pos_enc_dim = pos_enc_dim
        self.box_size = box_size
        self.lin1a.weight.data.fill_(0.0)
        # self.act = nn.ReLU()

    def forward(self, input_list):
        z = input_list[1]
        consn = input_list[0]
        batch_size = z[0].shape[0]
        cons_pos = consn
        # cons_pos = positional_encoding_geom2(consn,self.pos_enc_dim,self.box_size)
        if len(cons_pos.shape) < 3:
            posi_n = torch.stack(batch_size * [cons_pos], 0)
        else:
            posi_n = cons_pos
        conf_feat = torch.stack(posi_n.shape[1] * [z[0]], 1).squeeze()
        res = self.act(self.lin0(torch.cat([posi_n, conf_feat], 2)))
        res = self.act(self.lin1(res))
        res = self.lin1a(res)
        res = consn + 1 / self.n_layers * res
        return [res, z]


class LinearBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBlock, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim, bias=False)
        self.act = nn.ELU()

    def forward(self, x):
        res = self.act(self.lin(x))
        return res
