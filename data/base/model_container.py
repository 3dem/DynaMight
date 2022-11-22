#!/usr/bin/env python3
import warnings
from typing import List, TypeVar, Any, Tuple, Dict, Union
import torch
import os
import shutil
import numpy as np
import torch.nn.functional as F

Tensor = TypeVar('torch.tensor')
shape_type = Union[torch.Size, List[int], Tuple[int, ...]]


class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


def get_activation_function_by_name(name):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "elu":
        return torch.nn.ELU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    if name == "leaky_relu":
        return torch.nn.LeakyReLU()
    if name == "sin":
        return Sin()
    raise RuntimeError("Activation function not supported")


class Weights(torch.nn.Module):
    def __init__(
            self,
            shape: shape_type,
            bias: bool,
            fill_value: float
    ) -> None:
        super(Weights, self).__init__()
        self.weight = torch.nn.Parameter(torch.full(shape, float(fill_value), requires_grad=True))
        self.do_bias = bias
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, _) -> Tensor:
        if self.do_bias:
            return self.weight + self.bias
        else:
            return self.weight


class BaseMLP(torch.nn.Module):
    def __init__(self, definition: List[Union[int, str]]) -> None:
        super(BaseMLP, self).__init__()

        for i in np.arange(1, len(definition)):
            if (i % 2 == 1 and isinstance(definition[i], str)) or \
                    (i % 2 == 0 and not isinstance(definition[i], str)):
                raise RuntimeError("Bad network definition")

        layers = [torch.nn.Linear(definition[0], definition[1])]
        last_layer_size = definition[1]
        self.final_linear = layers[0]
        if len(definition) > 2:
            for i in np.arange(2, len(definition)):
                v = definition[i]
                if isinstance(v, str):
                    layers.append(get_activation_function_by_name(v))
                else:
                    self.final_linear = torch.nn.Linear(last_layer_size, v)
                    layers.append(self.final_linear)
                    last_layer_size = v

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(x)


class ResidBlock(torch.nn.Module):
    def __init__(self, dim: int, activation_fn, normalize: bool = False, init_factor: float = 1.) -> None:
        super(ResidBlock, self).__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.linear.weight.data *= init_factor
        self.linear.bias.data *= init_factor
        self.activation_fn = activation_fn
        self.norm = torch.nn.LayerNorm(dim) if normalize else None

    def forward(self, x):
        if self.norm is None:
            return self.activation_fn(self.linear(x) + x)
        else:
            return self.activation_fn(self.norm(self.linear(x)) + x)


class ResidMLP(torch.nn.Module):
    def __init__(
            self,
            resid_dim: int,
            resid_count: int,
            output_dim: int,
            activation,
            input_dim: int = None,
            normalize: bool = False,
            init_factor: float = 1.
    ) -> None:
        super(ResidMLP, self).__init__()

        if input_dim is not None:
            layers = [torch.nn.Linear(input_dim, resid_dim), activation]
        else:
            layers = []
        for i in range(resid_count):
            layers.append(ResidBlock(resid_dim, activation, normalize, init_factor))
        self.final_linear = torch.nn.Linear(resid_dim, output_dim)
        layers.append(self.final_linear)

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(x)


class ResidCnnBlock(torch.nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation_fn = activation

    def forward(self, x):
        return self.activation_fn(self.conv(x) + x)


class ResidCNN(torch.nn.Module):
    def __init__(self, input_dim, resid_dim, resid_count, output_dim, activation):
        super().__init__()
        layers = [torch.nn.Conv2d(input_dim, resid_dim, kernel_size=3, padding=1)]
        for i in range(resid_count):
            layers.append(ResidCnnBlock(resid_dim, activation))
        self.final_linear = torch.nn.Conv2d(resid_dim, output_dim, kernel_size=3, padding=1)
        layers.append(self.final_linear)
        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class ModelContainer(torch.nn.Module):
    def __init__(
            self,
            definition: Dict[str, Union[int, str, List, bool, shape_type]]
    ):
        """
        Example:
        definition={
            'type': 'Weights',
            'shape': [256, 256],
            'bias': True,
            'fill_value': 1
        }
        definition={
            'type': 'BaseMLP',
            'layers': [10, 256, 'elu', 256, 'elu', 128, 'elu', 64, 'elu', LATENT_SIZE*2]
        }
        definition={
            'type': 'ResidMLP',
            'input_dim': 10,
            'resid_dim': 256,
            'resid_count': 5,
            'output_dim': 1,
            'activation': 'relu'
        }
        definition={
            'type': 'ResidCNN',
            'input_dim': 3,
            'resid_dim': 256,
            'resid_count': 5,
            'output_dim': 1,
            'activation': 'relu'
        }
        """
        super().__init__()
        if definition['type'] == "Weights":
            self.model = Weights(
                shape=definition['shape'],
                bias=definition['bias'] if 'bias' in definition else False,
                fill_value=definition['fill_value'] if 'fill_value' in definition else 0.
            )
            self.input_dim = None
            self.output_dim = definition['shape']
        elif definition['type'] == "Linear":
            self.model = BaseMLP([definition['input_dim'], definition['output_dim']])
            self.input_dim = definition['input_dim']
            self.output_dim = definition['output_dim']
        elif definition['type'] == "BaseMLP":
            self.model = BaseMLP(definition['layers'])
            self.input_dim = definition['layers'][0] if not isinstance(definition['layers'][0], str) \
                else definition['layers'][1]
            self.output_dim = definition['layers'][-1] if not isinstance(definition['layers'][-1], str) \
                else definition['layers'][-2]
        elif definition['type'] == "ResidMLP":
            self.model = ResidMLP(
                input_dim=definition['input_dim'] if 'input_dim' in definition else None,
                resid_dim=definition['resid_dim'],
                resid_count=definition['resid_count'],
                output_dim=definition['output_dim'],
                activation=get_activation_function_by_name(definition['activation']),
                normalize='normalize' in definition and definition['normalize'],
                init_factor=definition['init_factor'] if 'init_factor' in definition else 1.
            )
            self.input_dim = definition['input_dim'] if 'input_dim' in definition else definition['resid_dim']
            self.output_dim = definition['output_dim']
        elif definition['type'] == "ResidCNN":
            self.model = ResidCNN(
                input_dim=definition['input_dim'],
                resid_dim=definition['resid_dim'],
                resid_count=definition['resid_count'],
                output_dim=definition['output_dim'],
                activation=get_activation_function_by_name(definition['activation'])
            )
            self.input_dim = definition['input_dim']
            self.output_dim = definition['output_dim']
        else:
            raise RuntimeError(f"Unknown network type '{definition['type']}'")

        self.definition = definition

    def forward(self, x):
        return self.model(x)

    def get_state_dict(self) -> Dict:
        return {
            "type": "NetworkContainer",
            "version": "0.0.1",
            "model": self.model.state_dict(),
            "definition": self.definition,
        }

    def save_to_file(self, path: str) -> None:
        state_dict = self.get_state_dict()
        path_backup = path + "_backup"
        torch.save(state_dict, path_backup)
        os.replace(path_backup, path)

    @staticmethod
    def load_from_state_dict(state_dict):
        if "type" not in state_dict or state_dict["type"] != "NetworkContainer":
            raise TypeError("Input is not an 'NetworkContainer' instance.")
        if "version" not in state_dict:
            raise RuntimeError("NetworkContainer instance lacks version information.")
        if state_dict["version"] == "0.0.1":
            container = ModelContainer(definition=state_dict["definition"])
            container.model.load_state_dict(state_dict["model"])
            return container
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")

    @staticmethod
    def load_from_file(path: str, map_location="cpu"):
        path_backup = path + "_backup"
        try:  # Try loading main file
            state_dict = torch.load(path, map_location=map_location)
        except Exception:  # Except all and try again with backup
            warnings.warn("Failed to load main checkpoint file. will try to load backup file instead.")
            state_dict = torch.load(path_backup, map_location=map_location)
        container = ModelContainer.load_from_state_dict(state_dict)
        return container


class PolynomialModelWrapper(torch.nn.Module):
    def __init__(self, model, output_size, orders: List[Union[float, int]] = None):
        super(PolynomialModelWrapper, self).__init__()
        self.model = model
        self.output_size = output_size

        if orders is not None and len(orders) != self.model.output_dim:
            raise RuntimeError("Orders count must match model output")

        x = torch.arange(self.output_size, dtype=torch.float32) / self.output_size

        if orders is None:
            self.range = torch.zeros([self.model.output_dim, output_size])
            for i in range(self.model.output_dim):
                self.range[i] = torch.pow(x, i)
        else:
            self.range = torch.zeros([len(orders), output_size])
            for i in range(len(orders)):
                self.range[i] = torch.pow(x, orders[i])

    def forward(self, input):
        self.range = self.range.to(input.device)
        c = self.model(input)
        output = torch.zeros([input.shape[0], self.output_size]).to(input.device)
        for i in range(self.range.shape[0]):
            output += c[:, i, None] * self.range[i][None, :]
        return output


class PiecewiseLinearModelWrapper(torch.nn.Module):
    def __init__(self, model, output_size):
        super(PiecewiseLinearModelWrapper, self).__init__()
        self.model = model
        self.order = self.model.output_dim
        self.output_size = output_size
        if self.order % 2 == 0:
            raise RuntimeError("Network output size must be odd.")

        self.range = torch.linspace(-1., 1., self.output_size, dtype=torch.float32)

    def forward(self, input):
        self.range = self.range.to(input.device)
        c = self.model(input)
        output = torch.zeros([input.shape[0], self.output_size]).to(input.device)
        output += c[:, 0, None]
        for i in range((self.order - 1) // 2):
            j = 2*i + 1
            output += F.relu(c[:, j, None] * self.range[None, :] + c[:, j+1, None])
        return output


class PowerModelWrapper(torch.nn.Module):
    def __init__(self, model, exp: bool = True, square: bool = True):
        super(PowerModelWrapper, self).__init__()
        self.exp = exp
        self.square = square
        self.model = model

    def forward(self, x):
        if self.exp and self.square:
            return torch.square(torch.exp(self.model(x)) - 1)
        elif self.exp:
            return torch.exp(self.model(x)) - 1
        elif self.square:
            return torch.square(self.model(x))
        else:
            return self.model(x)

    def get_loss(self, input, target, weight=None):
        self.model.train()

        if self.exp and self.square:
            target = torch.log(torch.sqrt(target) + 1)
        elif self.exp:
            target = torch.log(target + 1)
        elif self.square:
            target = torch.sqrt(target)

        if weight is None:
            return F.mse_loss(target, self.model(input))
        else:
            return torch.sum(weight * torch.square(target - self.model(input))) / torch.sum(weight)


class RadialBasisFunctions1D(torch.nn.Module):
    def __init__(self, size, strides, width=1.):
        super().__init__()
        x = torch.arange(size)
        mus = torch.linspace(0, size - 1, int(np.ceil(size / strides)))
        s = mus[1] - mus[0]
        w = s * width
        basis = torch.exp(-torch.square((x[:, None] - mus[None, :]) / w)) / w
        basis.requires_grad = False
        self.basis = torch.nn.Parameter(basis)

    def size(self):
        return self.basis.shape[-1]

    def forward(self, input):
        return torch.sum(input[:, None, :] * self.basis[None, ...], -1)
