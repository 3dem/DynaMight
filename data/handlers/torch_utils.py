#!/usr/bin/env python

"""
Simple Pytorch tools
"""

import argparse
import glob
import importlib.util
import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def standardize(np_input):
    mean = np.mean(np_input, axis=(1, 2, 3, 4))
    mean = np.resize(mean, (np_input.shape[0], 1, 1, 1, 1))
    std = np.std(np_input, axis=(1, 2, 3, 4)) + 1e-12
    std = np.resize(std, (np_input.shape[0], 1, 1, 1, 1))
    return mean, std


def torch_standardize(torch_input):
    mean = torch.mean(torch_input, dim=(1, 2, 3, 4))
    mean = torch.reshape(mean, (torch_input.shape[0], 1, 1, 1, 1))
    std = torch.std(torch_input, dim=(1, 2, 3, 4)) + 1e-12
    std = torch.reshape(std, (torch_input.shape[0], 1, 1, 1, 1))
    return mean, std


def normalize(np_input):
    norm = np.sqrt(np.sum(np.square(np_input), axis=(1, 2, 3, 4))) + 1e-12
    norm = np.resize(norm, (np_input.shape[0], 1, 1, 1, 1))
    return norm


def torch_normalize(torch_input):
    norm = torch.sqrt(torch.sum((torch_input) ** 2, dim=(1, 2, 3, 4))) + 1e-12
    norm = torch.reshape(norm, (torch_input.shape[0], 1, 1, 1, 1))
    return norm


def make_imshow_fig(data):
    if len(data.shape) == 3:
        data = data[data.shape[0] // 2]

    if type(data).__module__ == 'torch':
        data = data.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(data)
    plt.axis("off")
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    matplotlib.use(backend)

    return fig


def make_scatter_fig(x, y):
    if type(x).__module__ == 'torch':
        x = x.detach().data.cpu().numpy()
    if type(y).__module__ == 'torch':
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 7))
    alpha = min(10./np.sqrt(len(x)), 1.)
    ax.scatter(x, y, edgecolors=None, marker='.', c=np.arange(len(x)), cmap="summer", alpha=alpha)

    mx = np.mean(x)
    sx = np.std(x)*3
    my = np.mean(y)
    sy = np.std(y)*3

    ax.set_xlim([mx-sx, mx+sx])
    ax.set_ylim([my-sy, my+sy])

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    matplotlib.use(backend)
    return fig


def make_line_fig(x, y, y_log=False):
    if type(x).__module__ == 'torch':
        x = x.detach().data.cpu().numpy()
    if type(y).__module__ == 'torch':
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y)

    if y_log:
        ax.set_yscale('log')

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    matplotlib.use(backend)
    return fig


def make_series_line_fig(data, y_log=False):
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 5))
    for d in data:
        x = d['x']
        y = d['y']
        if type(x).__module__ == 'torch':
            x = x.detach().data.cpu().numpy()
        if type(y).__module__ == 'torch':
            y = y.detach().data.cpu().numpy()
        ax.plot(x, y,
                label=d['label'],
                color=d['color'] if 'color' in d else None,
                linestyle=d['linestyle'] if 'linestyle' in d else None
        )

    ax.legend()
    ax.grid()

    if y_log:
        ax.set_yscale('log')

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    matplotlib.use(backend)
    return fig


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def optimizer_set_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    # plt.barh(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.barh(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.yticks(range(0, len(ave_grads), 1), layers)
    plt.xlabel("average gradient")
    plt.ylabel("Layers")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)],
               ['max-gradient', 'mean-gradient'])
    plt.show()