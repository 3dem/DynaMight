from pathlib import Path
from typing import Optional
import torch
import numpy as np
import mrcfile
from ..data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
from ..data.dataloaders.relion import RelionDataset
from torch.utils.data import DataLoader
from ..utils.utils_new import pdb2points
from .utils import compute_dimensionality_reduction, compute_latent_space_and_colors, compute_max_deviation
from .visualizer import Visualizer_val, Visualizer
import starfile
from .._cli import cli
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



@cli.command()
def explore_latent_space(
    output_directory: Path,
    refinement_star_file: Optional[Path] = None,
    checkpoint_file: Optional[Path] = None,
    half_set: int = 1,
    mask_file: Optional[Path] = None,
    particle_diameter: Optional[float] = None,
    soft_edge_width: float = 20,
    batch_size: int = 100,
    gpu_id: Optional[int] = 0,
    preload_images: bool = True,
    n_workers: int = 8,
    dimensionality_reduction_method: str = 'PCA',
    inverse_deformation: Optional[str] = None,
    atomic_model: str = None,
    reduce_by_deformation: bool = False,
    cluster: bool = False
):
    # todo: @schwab implement preload images

    # load and prepare models for inference
    print('Visualizing latent space for half:', half_set)
    print('For visualization of the other half set, specify --half_set')

    device = "cpu" if gpu_id is None else 'cuda:' + str(gpu_id)
    if checkpoint_file is None:
        checkpoint_file = output_directory / \
            'forward_deformations/checkpoints/checkpoint_final.pth'
    cp = torch.load(checkpoint_file, map_location=device)
    if inverse_deformation is not None:
        cp_inv = torch.load(inverse_deformation, map_location=device)
        inv_half1 = cp_inv['inv_half1']
        inv_half2 = cp_inv['inv_half2']
        inv_half1.load_state_dict(cp_inv['inv_half1_state_dict'])
        inv_half2.load_state_dict(cp_inv['inv_half2_state_dict'])

    if refinement_star_file == None:
        refinement_star_file = cp['refinement_directory']
        if refinement_star_file.suffix == '.star':
            pass
        else:
            refinement_star_file = refinement_star_file / 'run_data.star'
        # refinement_star_file = Path(
        #    '/cephfs/aburt/dynamight-tests/stan-kmn/data/consensus_360px/particles.star')
        #refinement_star_file = refinement_star_file / 'run_data.star'

    dataframe = starfile.read(refinement_star_file)
    circular_mask_thickness = soft_edge_width
    if half_set == 0:
        encoder_h1 = cp['encoder_half1']
        decoder_h1 = cp['decoder_half1']
        encoder_h2 = cp['encoder_half2']
        decoder_h2 = cp['decoder_half2']
    else:
        encoder = cp['encoder_half'+str(half_set)]
        decoder = cp['decoder_half'+str(half_set)]

    poses = cp['poses']

    relion_dataset = RelionDataset(
        path=refinement_star_file.resolve(),
        circular_mask_thickness=soft_edge_width,
        particle_diameter=particle_diameter,
    )
    dataset = relion_dataset.make_particle_dataset()
    diameter_ang = relion_dataset.particle_diameter
    box_size = relion_dataset.box_size
    ang_pix = relion_dataset.pixel_spacing_angstroms

    if half_set == 0:
        encoder_h1.load_state_dict(
            cp['encoder_half1_state_dict'])
        decoder_h1.load_state_dict(
            cp['decoder_half1_state_dict'])
        encoder_h2.load_state_dict(
            cp['encoder_half2_state_dict'])
        decoder_h2.load_state_dict(
            cp['decoder_half2_state_dict'])
    else:
        encoder.load_state_dict(cp['encoder_half'+str(half_set)+'_state_dict'])
        decoder.load_state_dict(cp['decoder_half'+str(half_set)+'_state_dict'])


    poses.load_state_dict(cp['poses_state_dict'])

    '''Computing indices for the second half set'''

    if half_set == 1:
        indices = cp['indices_half1'].cpu().numpy()
    elif half_set == 2:
        inds_half1 = cp['indices_half1'].cpu().numpy()
        indices = np.asarray(
            list(set(range(len(dataset))) - set(list(inds_half1))))
    elif half_set == 0:
        indices = cp['indices_val'].cpu().numpy()


    if half_set == 0:
        decoder_h1.p2i.device = device
        decoder_h1.projector.device = device
        decoder_h1.image_smoother.device = device
        decoder_h1.p2v.device = device
        decoder_h1.device = device
        decoder_h1.to(device)


        decoder_h2.p2i.device = device
        decoder_h2.projector.device = device
        decoder_h2.image_smoother.device = device
        decoder_h2.p2v.device = device
        decoder_h2.device = device
        decoder_h2.to(device)
    else:
        decoder.p2i.device = device
        decoder.projector.device = device
        decoder.image_smoother.device = device
        decoder.p2v.device = device
        decoder.device = device
        decoder.to(device)

    if mask_file:
        with mrcfile.open(mask_file) as mrc:
            mask = torch.tensor(mrc.data)
            mask = mask.movedim(0, 2).movedim(0, 1)

    if atomic_model:
        pdb_pos = pdb2points(atomic_model) / (box_size * ang_pix) - 0.5

    dataset_half = torch.utils.data.Subset(dataset, indices)
    dataloader_half = DataLoader(
        dataset=dataset_half,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True
    )

    batch = next(iter(dataloader_half))

    data_preprocessor = ParticleImagePreprocessor()
    data_preprocessor.initialize_from_stack(
        stack=batch['image'],
        circular_mask_radius=diameter_ang / (2 * ang_pix),
        circular_mask_thickness=circular_mask_thickness / ang_pix)

    if half_set == 0:
        latent_dim = decoder_h1.latent_dim
        with torch.no_grad():
            latent_space_h1, latent_colors_h1, point_colors_h1, feature_vec_h1 = compute_latent_space_and_colors(
                encoder_h1, decoder_h1, dataloader_half, poses, data_preprocessor, indices, reduce_by_deformation
            )
            latent_space_h2, latent_colors_h2, point_colors_h2, feature_vec_h2 = compute_latent_space_and_colors(
                encoder_h2, decoder_h2, dataloader_half, poses, data_preprocessor, indices, reduce_by_deformation
            )

        if latent_space_h1.shape[1] > 2:
            print('Computing dimensionality reduction')
            if reduce_by_deformation is True:
                embedded_latent_space_h1 = compute_dimensionality_reduction(
                    feature_vec_h1, dimensionality_reduction_method)
                embedded_latent_space_h2 = compute_dimensionality_reduction(
                    feature_vec_h2, dimensionality_reduction_method)
            else:
                embedded_latent_space_h1 = compute_dimensionality_reduction(
                    latent_space_h1, dimensionality_reduction_method)
                embedded_latent_space_h2 = compute_dimensionality_reduction(
                    latent_space_h2, dimensionality_reduction_method)
            print('Dimensionality reduction finished')
        else:
            embedded_latent_space_h1 = latent_space_h1.cpu().numpy()
            embedded_latent_space_h2 = latent_space_h2.cpu().numpy()
        embedded_latent_space_h1 = torch.tensor(embedded_latent_space_h1)
        embedded_latent_space_h2 = torch.tensor(embedded_latent_space_h2)

        closest_idx = np.argmin(latent_colors_h1['amount'])
        latent_closest = embedded_latent_space_h1[closest_idx]
    else:
        latent_dim = decoder.latent_dim
        latent_space, latent_colors, point_colors, feature_vec = compute_latent_space_and_colors(
            encoder, decoder, dataloader_half, poses, data_preprocessor, indices, reduce_by_deformation
        )
        if latent_space.shape[1] > 2:
            print('Computing dimensionality reduction')
            if reduce_by_deformation is True:
                embedded_latent_space = compute_dimensionality_reduction(
                    feature_vec, dimensionality_reduction_method)
            else:
                embedded_latent_space = compute_dimensionality_reduction(
                    latent_space, dimensionality_reduction_method)
            print('Dimensionality reduction finished')
        else:

            if reduce_by_deformation is True:
                embedded_latent_space = compute_dimensionality_reduction(
                    feature_vec, dimensionality_reduction_method)
            else:
                embedded_latent_space = latent_space.cpu().numpy()
        embedded_latent_space = torch.tensor(embedded_latent_space)
        closest_idx = np.argmin(latent_colors['amount'])
        latent_closest = embedded_latent_space[closest_idx]


    r = torch.zeros([2, 3])
    t = torch.zeros([2, 2])
    if half_set == 0:
        cons_volume_h1 = decoder_h1.generate_consensus_volume()
        cons_volume_h1 = cons_volume_h1[0].detach().cpu().numpy()
        cons_volume_h2 = decoder_h2.generate_consensus_volume()
        cons_volume_h2 = cons_volume_h2[0].detach().cpu().numpy()
    else:
        cons_volume = decoder.generate_consensus_volume()
        cons_volume = cons_volume[0].detach().cpu().numpy()
    if half_set == 0:
        nap_cons_pos_h1 = (
            0.5 + decoder_h1.model_positions.detach().cpu()) * box_size
        nap_cons_pos_h2 = (
            0.5 + decoder_h2.model_positions.detach().cpu()) * box_size
        nap_zeros = torch.zeros(nap_cons_pos_h1.shape[0])
        nap_cons_pos_h1 = torch.cat(
            [nap_zeros.unsqueeze(1), nap_cons_pos_h1], 1)
        nap_cons_pos_h2 = torch.cat(
            [nap_zeros.unsqueeze(1), nap_cons_pos_h2], 1)
        nap_cons_pos_h1 = torch.stack(
            [nap_cons_pos_h1[:, 0], nap_cons_pos_h1[:, 3], nap_cons_pos_h1[:, 2], nap_cons_pos_h1[:, 1]], 1)
        nap_cons_pos_h2 = torch.stack(
            [nap_cons_pos_h2[:, 0], nap_cons_pos_h2[:, 3], nap_cons_pos_h2[:, 2], nap_cons_pos_h2[:, 1]], 1)
    else:
        nap_cons_pos = (
            0.5 + decoder.model_positions.detach().cpu()) * box_size
        nap_zeros = torch.zeros(nap_cons_pos.shape[0])
        nap_cons_pos = torch.cat([nap_zeros.unsqueeze(1), nap_cons_pos], 1)
        nap_cons_pos = torch.stack(
            [nap_cons_pos[:, 0], nap_cons_pos[:, 3], nap_cons_pos[:, 2], nap_cons_pos[:, 1]], 1)

    with torch.no_grad():
        if half_set == 0:
            V0_h1 = decoder_h1.generate_volume(torch.zeros(2, latent_dim).to(
                device), r.to(device), t.to(device)).float()
            V0_h2 = decoder_h2.generate_volume(torch.zeros(2, latent_dim).to(
                device), r.to(device), t.to(device)).float()
        else:
            V0 = decoder.generate_volume(torch.zeros(2, latent_dim).to(
                device), r.to(device), t.to(device)).float()

    if half_set == 0:
        max_diff, diff_col = compute_max_deviation(
            latent_space_h1, latent_space_h2, decoder_h1, decoder_h2)
        latent_colors_h1['difference'] = diff_col.cpu().numpy()
        latent_colors_h2['difference'] = diff_col.cpu().numpy()


    if cluster == True:

        kmeans = KMeans(n_clusters=80).fit(feature_vec.cpu().numpy())
        latent_colors['cluster'] = kmeans.labels_

        sortinds = np.argsort(kmeans.labels_)
        new_star = dataframe.copy()
        new_star['particles'] = dataframe['particles'].loc[list(
            sortinds)]


        #star_directory = output_directory / 'subsets'
        #star_directory.mkdir(exist_ok=True, parents=True)
        # starfile.write(new_star, star_directory /
        #              ('starfile_ordered_half' + str(half_set) + '.star'))

    if half_set == 0:

        vis = Visualizer_val(
            embedded_latent_space_h1,
            embedded_latent_space_h2,
            latent_space_h1,
            latent_space_h2,
            decoder_h1,
            decoder_h2,
            V0_h1,
            cons_volume_h1,
            nap_cons_pos_h1,
            point_colors_h1,
            latent_colors_h1,
            dataframe,
            indices,
            latent_closest,
            half_set,
            output_directory,
            max_diff
        )
        vis.run()
    else:
        vis = Visualizer(
            embedded_latent_space,
            latent_space,
            decoder,
            V0,
            cons_volume,
            nap_cons_pos,
            point_colors,
            latent_colors,
            dataframe,
            indices,
            latent_closest,
            half_set,
            output_directory
        )
        vis.run()
