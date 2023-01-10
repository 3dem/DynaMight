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


def initialize_points_from_binary_volume(
    volume: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """Place points randomly within a binary volume."""
    points = []
    sidelength = volume.shape[0]
    while len(points) < n_points:
        random_points = torch.FloatTensor(
            n_points, 3).uniform_(0, sidelength-1)
        idx = torch.round(random_points).long()
        valid_points = volume[idx[:, 0], idx[:, 1], idx[:, 2]] == 1
        random_points /= (sidelength - 1)  # [0, 1]
        random_points -= 0.5  # [-0.5, 0.5]
        random_points = random_points[valid_points]
        if len(points) > 0:
            points = torch.cat([points, random_points], dim=0)
        else:
            points = random_points
    return points[:n_points]


def initialize_points_from_volume(
    volume: torch.Tensor,
    threshold: float,
    n_points: int,
) -> torch.Tensor:
    """Place points randomly within a volume in regions above a given threshold."""
    return initialize_points_from_binary_volume(volume > threshold, n_points)
