import torch


def positional_encoding(
    coords: torch.Tensor, enc_dim: int, DD: int
) -> torch.Tensor:
    """Positional encoding of continuous coordinates.

    TODO: figure out what this does
    RECOMMENDATION AB: enc_dim should be `output_dim` and match output dimensionality

    Parameters
    ----------
    coords: torch.Tensor
        `(n, 3)` array of input coordinates.
    enc_dim: int

    DD

    Returns
    -------

    """
    D2 = DD // 2
    freqs = torch.arange(enc_dim, dtype=torch.float).to(coords.device)
    freqs = D2 * (1. / D2) ** (freqs / (enc_dim - 1))  # 2/D*2pi to 2pi
    freqs = freqs.view(*[1] * len(coords.shape), -1)
    coords = coords.unsqueeze(-1)
    k = coords * freqs
    s = torch.sin(k)
    c = torch.cos(k)
    x = torch.cat([s, c], -1)
    return x.flatten(-2)
