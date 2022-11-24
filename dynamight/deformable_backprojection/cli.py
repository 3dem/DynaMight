from pathlib import Path
from typing import Optional

from typer import Typer, Option

cli = Typer(add_completion=False)

@cli.command()
def deformable_backprojection(
    refinement_star_file: Path,
    output_directory: Path,
    mask_file: Path = Option(None),
    ctf: bool = Option(True),
    gpu_id: Optional[int] = Option(None),
    batch_size: int = Option(24),
    preload_images: bool = Option(False),
    pooling_fraction: Optional[float] = Option(0.05),
    pooling_multiplier: Optional[float] = Option(3),
    particle_diameter: Optional[float] = Option(None),
    mask_soft_edge_width: int = Option(20),
):
    print(refinement_star_file)


@cli.command()
def learn_inverse_deformation_field():
    pass