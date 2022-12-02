import os
import sys
import time
from pathlib import Path
from typing import Optional


import torch
import argparse
import numpy as np
import mrcfile
import torch.nn.functional as F
from ..data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
from torch.utils.data import DataLoader

from ..utils.utils_new import initialize_dataset, pdb2points, bezier_curve, make_equidistant

import matplotlib.pylab as plt
import napari
from tsnecuda import TSNE
from sklearn.decomposition import PCA
import umap
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from magicgui import widgets
from matplotlib.widgets import LassoSelector, PolygonSelector
import starfile
#from matplotlib.path import Path
from tqdm import tqdm

from typer import Option

from .._cli import cli


@cli.command()
def explore_latent_space(
    refinement_star_file: Path,
    checkpoint_file: Path,
    output_directory: Path,
    half_set: int = 1,
    mask_file: Optional[Path] = None,
    particle_diameter: Optional[float] = None,
    soft_edge_width: float = 20,
    batch_size: int = 100,
    gpu_id: int = 0,
    preload_images: bool = True,
    n_workers: int = 8,
    dimensionality_reduction_method: str = 'PCA',
    inverse_deformation: Optional[str] = None,
    atomic_model: str = None,
):

    dataframe = starfile.read(refinement_star_file)
    circular_mask_thickness = soft_edge_width

    device = "cpu" if gpu_id == "-1" else 'cuda:' + str(gpu_id)

    '-----------------------------------------------------------------------------'
    'load and prepare models for inference'
    cp = torch.load(checkpoint_file, map_location=device)
    if inverse_deformation != None:
        cp_inv = torch.load(inverse_deformation, map_location=device)
        inv_half1 = cp_inv['inv_half1']
        inv_half2 = cp_inv['inv_half2']
        inv_half1.load_state_dict(cp_inv['inv_half1_state_dict'])
        inv_half2.load_state_dict(cp_inv['inv_half2_state_dict'])

    encoder = cp['encoder_' + 'half' + str(half_set)]
    cons_model = cp['consensus']

    decoder = cp['decoder_' + 'half' + str(half_set)]
    poses = cp['poses']

    dataset, diameter_ang, box_size, ang_pix, optics_group = initialize_dataset(refinement_star_file,
                                                                                soft_edge_width,
                                                                                preload_images,
                                                                                particle_diameter)
    try:
        cons_model.vol_box
        decoder.vol_box
    except:
        cons_model.vol_box = box_size
        decoder.vol_box = box_size

    encoder.load_state_dict(
        cp['encoder_' + 'half' + str(half_set) + '_state_dict'])
    cons_model.load_state_dict(cp['consensus_state_dict'])
    decoder.load_state_dict(
        cp['decoder_' + 'half' + str(half_set) + '_state_dict'])
    poses.load_state_dict(cp['poses_state_dict'])
    if half_set == 1:
        # indices = np.arange(len(dataset))
        indices = cp['indices_half1'].cpu().numpy()
    else:
        inds_half1 = cp['indices_half1'].cpu().numpy()
        indices = np.asarray(
            list(set(range(len(dataset))) - set(list(inds_half1))))

    n_classes = 1  # ??????

    points = cons_model.pos.detach().cpu()

    points = torch.tensor(points)
    cons_model.pos = torch.nn.Parameter(points, requires_grad=False)
    cons_model.n_points = len(points)
    decoder.n_points = cons_model.n_points

    cons_model.p2i.device = device
    decoder.p2i.device = device
    cons_model.proj.device = device
    decoder.proj.device = device
    cons_model.i2F.device = device
    decoder.i2F.device = device
    cons_model.p2v.device = device
    decoder.p2v.device = device
    decoder.device = device
    cons_model.device = device
    decoder.to(device)
    cons_model.to(device)

    if mask_file:
        with mrcfile.open(mask_file) as mrc:
            mask = torch.tensor(mrc.data)
            mask = mask.movedim(0, 2).movedim(0, 1)

    max_diameter_ang = box_size * \
        optics_group['pixel_size'] - circular_mask_thickness

    if particle_diameter is None:
        diameter_ang = box_size * 1 * \
            optics_group['pixel_size'] - circular_mask_thickness
        print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
    else:
        if particle_diameter > max_diameter_ang:
            print(
                f"WARNING: Specified particle diameter {round(particle_diameter)} angstrom is too large\n"
                f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
            )
            diameter_ang = max_diameter_ang
        else:
            diameter_ang = particle_diameter

    if preload_images:
        dataset.preload_images()

    if atomic_model:
        pdb_pos = pdb2points(atomic_model) / (box_size * ang_pix) - 0.5

    dataset_half1 = torch.utils.data.Subset(dataset, indices)
    data_loader_half1 = DataLoader(
        dataset=dataset_half1,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True)

    batch = next(iter(data_loader_half1))

    data_preprocessor = ParticleImagePreprocessor()
    data_preprocessor.initialize_from_stack(
        stack=batch['image'],
        circular_mask_radius=diameter_ang / (2 * optics_group['pixel_size']),
        circular_mask_thickness=circular_mask_thickness / optics_group['pixel_size'])

    box_size = optics_group['image_size']
    ang_pix = optics_group['pixel_size']

    pixel_distance = 1 / box_size

    latent_dim = decoder.latent_dim

    '-----------------------------------------------------------------------------'
    'Coloring by pose and shift'
    ccp = poses.orientations[indices]
    ccp = F.normalize(ccp, dim=1)
    ccp = ccp.detach().cpu().numpy()
    ccp = ccp / 2 + 0.5
    ccp = ccp[:, 2]

    cct = poses.translations[indices]
    cct = torch.linalg.norm(cct, dim=1)
    cct = cct.detach().cpu().numpy()
    cct = cct - np.min(cct)
    cct /= np.max(cct)

    '-----------------------------------------------------------------------------'
    'Evaluate model on the halfset'

    globaldis = torch.zeros(cons_model.pos.shape[0]).to(device)

    with torch.no_grad():
        for batch_ndx, sample in enumerate(tqdm(data_loader_half1)):
            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r, t = poses(idx)
            y = data_preprocessor.apply_square_mask(y)
            y = data_preprocessor.apply_translation(y, -t[:, 0], -t[:, 1])
            y = data_preprocessor.apply_circular_mask(y)
            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])
            y, r, ctf, t = y.to(device), r.to(
                device), ctf.to(device), t.to(device)
            mu, _ = encoder(y, ctf)
            mu_in = [mu]
            _, _, _, _, dis = decoder(mu_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                      cons_model.ampvar.to(device), t)
            d = torch.linalg.vector_norm(dis, dim=-1)
            dd = torch.mean(d, 0)
            globaldis += dd
            cons = torch.stack(d.shape[0] * [cons_model.pos], 0)
            dps = torch.sum(d > 2 * pixel_distance, 1)
            if batch_ndx == 0:
                z = mu
                c1 = torch.sum(d > pixel_distance, 1)
                c2 = torch.sum(d, 1)
                c3 = torch.mean(dis.movedim(2, 0) * (d > pixel_distance), -1)
                c4 = torch.sum(cons.movedim(2, 0) *
                               ((d > 2 * pixel_distance) * d), 2) / dps[None, :]

            else:
                z = torch.cat([z, mu])
                c1 = torch.cat([c1, torch.sum(d > pixel_distance, 1)])
                c2 = torch.cat([c2, torch.sum(d, 1)])
                c3 = torch.cat(
                    [c3, torch.mean(dis.movedim(2, 0) * (d > pixel_distance), -1)], 1)
                c4 = torch.cat([c4,
                                torch.sum(cons.movedim(2, 0) * ((d > 2 * pixel_distance) * d), 2) / dps[
                                    None:]],
                               1)

    c3 = torch.movedim(c3, 0, 1)
    c4 = torch.movedim(c4, 0, 1)
    c3 = F.normalize(c3, dim=1)
    c3 = c3 / 2 + 0.5

    closest_idx = torch.argmin(c2)

    z_lat = z[:, :latent_dim]
    dd = globaldis / torch.max(globaldis)
    dd = dd.cpu().numpy()

    if z.shape[1] > 2:
        print('Computing dimensionality reduction')
        if dimensionality_reduction_method == 'TSNE':
            zz = TSNE(perplexity=1000.0, num_neighbors=1000,
                      device=0).fit_transform(z.cpu())
        elif dimensionality_reduction_method == 'UMAP':
            zz = umap.UMAP(random_state=12, n_neighbors=100,
                           min_dist=1.0).fit_transform(z.cpu().numpy())
        elif dimensionality_reduction_method == 'PCA':
            pca = PCA(n_components=8)
            pca.fit(z.cpu().numpy())
            ex_var = pca.explained_variance_ratio_
            comps = pca.components_
            zz = PCA(n_components=2).fit_transform(z.cpu().numpy())
        print('Dimensionality reduction finished')
    else:
        zz = z.cpu().numpy()

    cc1 = c1.cpu().numpy()
    cc2 = c2.cpu().numpy()
    cc2 = cc2 / np.max(cc2)
    cc3 = c3.cpu().numpy()
    cc4 = c4.cpu().numpy()

    cc5 = cc4 / np.max(np.linalg.norm(cc4, axis=1)) / 2 + 0.5
    cc4b = np.linalg.norm(cc4, axis=1)
    cc5 = cc5 - np.min(cc5)
    cc5 = cc5 / np.max(cc5)

    cons0 = cons_model.pos.detach().cpu().numpy()
    c_cons = cons0 / (2 * np.max(np.abs(cons0))) + 0.5

    latent = zz * 100
    latent_closest = zz[closest_idx]
    latent_properties = {
        'region': cc2
    }

    r = torch.zeros([2, 3])
    t = torch.zeros([2, 2])
    cons_volume = cons_model.volume(r.to(device), t.to(device))
    cons_volume = cons_volume[0].detach().cpu().numpy()

    nap_cons_pos = (0.5 + cons_model.pos.detach().cpu()) * box_size
    nap_zeros = torch.zeros(nap_cons_pos.shape[0])
    nap_cons_pos = torch.cat([nap_zeros.unsqueeze(1), nap_cons_pos], 1)
    nap_cons_pos = torch.stack(
        [nap_cons_pos[:, 0], nap_cons_pos[:, 3], nap_cons_pos[:, 2], nap_cons_pos[:, 1]], 1)

    with torch.no_grad():
        V0 = decoder.volume([torch.zeros(2, latent_dim).to(device)], r.to(device),
                            cons_model.pos.to(
                                device), cons_model.amp.to(device),
                            cons_model.ampvar.to(device), t.to(device)).float()

    amps = cons_model.amp.detach().cpu()
    amps = amps[0]
    amps -= torch.min(amps)
    amps /= torch.max(amps)

    lat_colors = {'amount': cc1, 'direction': cc3, 'location': cc5, 'index': indices, 'pose': ccp,
                  'shift': cct}

    class Visualizer:
        def __init__(self, z, latent, decoder, cons_model, V0, cons_volume, nap_cons_pos, dd, amps,
                     lat_cols, starfile, indices, latent_closest):
            self.z = z
            self.latent = latent
            self.vol_viewer = napari.Viewer(ndisplay=3)
            self.star = starfile
            self.decoder = decoder
            self.cons_model = cons_model
            self.cons_volume = cons_volume / np.max(cons_volume)
            self.V0 = (V0[0] / torch.max(V0)).cpu().numpy()
            self.vol_layer = self.vol_viewer.add_image(self.V0, rendering='iso', colormap='gray',
                                                       iso_threshold=0.15)
            self.cons_layer = self.vol_viewer.add_image(self.cons_volume, rendering='iso',
                                                        colormap='gray', iso_threshold=0.15)
            self.point_properties = {'activity': dd, 'amplitude': amps,
                                     'width': torch.nn.functional.softmax(cons_model.ampvar, 0)[
                                         0].detach().cpu()}
            self.nap_cons_pos = nap_cons_pos
            self.nap_zeros = torch.zeros(nap_cons_pos.shape[0])
            self.point_layer = self.vol_viewer.add_points(self.nap_cons_pos,
                                                          properties=self.point_properties, ndim=4,
                                                          size=1.5, face_color='activity',
                                                          face_colormap='jet', edge_width=0,
                                                          visible=False)
            self.cons_point_layer = self.vol_viewer.add_points(self.nap_cons_pos,
                                                               properties=self.point_properties, ndim=4,
                                                               size=1.5, face_color='activity',
                                                               face_colormap='jet', edge_width=0,
                                                               visible=False)
            plt.style.use('dark_background')
            self.latent_closest = latent_closest
            self.fig = Figure(figsize=(5, 5))
            self.lat_canv = FigureCanvas(self.fig)
            self.axes1 = self.lat_canv.figure.subplots()
            self.axes1.scatter(z[:, 0], z[:, 1], c=cc3)
            self.axes1.scatter(
                self.latent_closest[0], self.latent_closest[1], c='r')
            self.lat_im = self.vol_viewer.window.add_dock_widget(self.lat_canv, area='right',
                                                                 name='latent space')
            self.vol_viewer.window._qt_window.resizeDocks(
                [self.lat_im], [800], Qt.Horizontal)
            self.rep_menu = widgets.ComboBox(
                label='3d representation', choices=['volume', 'points'])
            self.rep_widget = widgets.Container(widgets=[self.rep_menu])
            self.repw = self.vol_viewer.window.add_dock_widget(self.rep_widget, area='bottom',
                                                               name='3d representation')
            self.rep_col_menu = widgets.ComboBox(label='gaussian colors',
                                                 choices=['displacement', 'amplitude', 'width'])
            self.rep_col_widget = widgets.Container(
                widgets=[self.rep_col_menu])
            self.repc = self.vol_viewer.window.add_dock_widget(self.rep_col_widget, area='bottom',
                                                               name='gaussian colors')
            self.rep_action_menu = widgets.ComboBox(label='action',
                                                    choices=['click', 'trajectory', 'particle number',
                                                             'starfile'])
            self.rep_action_widget = widgets.Container(
                widgets=[self.rep_action_menu])
            self.repa = self.vol_viewer.window.add_dock_widget(self.rep_action_widget, area='bottom',
                                                               name='action')
            self.color_menu = widgets.ComboBox(label='color by',
                                               choices=['direction', 'amount', 'location', 'density',
                                                        'log-density', 'index', 'pose', 'shift'])
            self.color_widget = widgets.Container(widgets=[self.color_menu])
            self.colw = self.vol_viewer.window.add_dock_widget(self.color_widget, area='right',
                                                               name='colors')
            self.lat_cols = lat_cols
            self.color_menu.changed.connect(self.coloring_changed)
            self.rep_col_menu.changed.connect(self.coloring_gaussians)
            self.rep_action_menu.changed.connect(self.action_changed)
            self.r = torch.zeros([2, 3])
            self.t = torch.zeros([2, 2])
            self.line = {'color': 'white',
                         'linewidth': 8, 'alpha': 1}

            self.nap_z = torch.zeros(nap_cons_pos.shape[0])
            self.cid = self.fig.canvas.mpl_connect(
                'button_press_event', self.onclick)
            self.lasso = LassoSelector(ax=self.axes1, onselect=self.onSelect,
                                       props=self.line, button=2, useblit=True)
            self.lasso.disconnect_events()
            self.star_nr = 0
            self.indices = indices
            self.poly = PolygonSelector(ax=self.axes1, onselect=self.get_part_nr, props=self.line,
                                        useblit=True)
            self.poly.disconnect_events()

        def run(self):
            napari.run()

        def coloring_changed(self, event):
            """This is a callback that update the current properties on the Shapes layer
            when the label menu selection changes
            """

            selected_label = event

            if selected_label == 'amount':
                self.axes1.clear()
                self.axes1.scatter(
                    self.z[:, 0], self.z[:, 1], c=self.lat_cols['amount'], alpha=0.1)
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()

            elif selected_label == 'direction':
                self.axes1.clear()
                self.axes1.scatter(
                    self.z[:, 0], self.z[:, 1], c=self.lat_cols['direction'], alpha=0.1)
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()

            elif selected_label == 'location':
                self.axes1.clear()
                self.axes1.scatter(
                    self.z[:, 0], self.z[:, 1], c=self.lat_cols['location'], alpha=0.1)
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()
            elif selected_label == 'density':
                h, x, y, p = plt.hist2d(
                    self.z[:, 0], self.z[:, 1], bins=(50, 50), cmap='hot')
                ex = [self.axes1.dataLim.x0, self.axes1.dataLim.x1, self.axes1.dataLim.y0,
                      self.axes1.dataLim.y1]
                self.axes1.clear()
                # axes1.hist2d(zz[:,0],zz[:,1],bins = (100,100), cmap = 'hot')
                self.axes1.imshow(
                    np.rot90(h), interpolation='gaussian', extent=ex, cmap='hot')
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()
            elif selected_label == 'log-density':
                h, x, y, p = plt.hist2d(
                    self.z[:, 0], self.z[:, 1], bins=(50, 50), cmap='hot')
                ex = [self.axes1.dataLim.x0, self.axes1.dataLim.x1, self.axes1.dataLim.y0,
                      self.axes1.dataLim.y1]
                self.axes1.clear()
                # axes1.hist2d(zz[:,0],zz[:,1],bins = (100,100), cmap = 'hot')
                self.axes1.imshow(np.log(1 + np.rot90(h)), interpolation='gaussian', extent=ex,
                                  cmap='hot')
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()
            elif selected_label == 'index':
                self.axes1.clear()
                self.axes1.scatter(
                    self.z[:, 0], self.z[:, 1], c=self.lat_cols['index'])
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()
            elif selected_label == 'pose':
                self.axes1.clear()
                self.axes1.scatter(zz[:, 0], zz[:, 1], c=self.lat_cols['pose'])
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()
            elif selected_label == 'shift':
                self.axes1.clear()
                self.axes1.scatter(
                    self.z[:, 0], self.z[:, 1], c=self.lat_cols['shift'])
                self.axes1.scatter(
                    self.latent_closest[0], self.latent_closest[1], c='r')
                self.lat_canv.draw()

        def coloring_gaussians(self, event):
            """This is a callback that update the current properties on the Shapes layer
            when the label menu selection changes
            """
            selected_label = event

            if selected_label == 'displacement':
                self.point_layer.face_color = 'activity'
                self.point_layer.refresh()
            elif selected_label == 'amplitude':
                self.point_layer.face_color = 'amplitude'
                self.point_layer.refresh()
            elif selected_label == 'width':
                self.point_layer.face_color = 'width'
                self.point_layer.refresh()

        def action_changed(self, event):
            """This is a callback that update the current properties on the Shapes layer
            when the label menu selection changes
            """
            selected_label = event

            if selected_label == 'click':
                self.cid = self.fig.canvas.mpl_connect(
                    'button_press_event', self.onclick)
                self.lasso.disconnect_events()
                self.poly.disconnect_events()
                self.lat_canv.draw_idle()
            elif selected_label == 'trajectory':
                self.fig.canvas.mpl_disconnect(self.cid)
                self.lasso = LassoSelector(ax=self.axes1, onselect=self.onSelect,
                                           props=self.line, button=2, useblit=True)

            elif selected_label == 'particle number':
                self.fig.canvas.mpl_disconnect(self.cid)
                self.lasso.disconnect_events()
                self.lat_canv.draw_idle()
                self.poly = PolygonSelector(ax=self.axes1, onselect=self.get_part_nr, props=self.line,
                                            useblit=True)

            elif selected_label == 'starfile':
                self.fig.canvas.mpl_disconnect(self.cid)
                self.lasso.disconnect_events()
                self.poly = PolygonSelector(ax=self.axes1, onselect=self.save_starfile, props=self.line,
                                            useblit=True)
                self.lat_canv.draw_idle()

        def onclick(self, event):
            global ix, iy
            ix, iy = event.xdata, event.ydata
            if self.latent.shape[1] > 2:
                inst_coord = torch.tensor(np.array([ix, iy])).float()
                dist = torch.linalg.norm(torch.tensor(
                    self.z) - inst_coord.unsqueeze(0), dim=1)
                inst_ind = torch.argmin(dist, 0)
                lat_coord = self.latent[inst_ind]
            else:
                inst_coord = torch.tensor(np.array([ix, iy])).float()
                dist = torch.linalg.norm(torch.tensor(
                    self.z) - inst_coord.unsqueeze(0), dim=1)
                inst_ind = torch.argmin(dist, 0)
                lat_coord = self.z[inst_ind]

            lat = torch.stack(2 * [torch.tensor(lat_coord)], 0)

            if self.rep_menu.current_choice == 'volume':
                vol = self.decoder.volume([lat.to(device)], self.r.to(device),
                                          self.cons_model.pos.to(device),
                                          self.cons_model.amp.to(device),
                                          self.cons_model.ampvar.to(device), self.t.to(device)).float()
                self.vol_layer.data = (
                    vol[0] / torch.max(vol[0])).detach().cpu().numpy()
            elif self.rep_menu.current_choice == 'points':
                proj, proj_im, proj_pos, pos, dis = self.decoder.forward([lat.to(device)],
                                                                         self.r.to(
                    device),
                    self.cons_model.pos.to(
                    device),
                    self.cons_model.amp.to(
                    device),
                    self.cons_model.ampvar.to(
                    device), self.t.to(device))
                p = torch.cat([self.nap_zeros.unsqueeze(1), (0.5 + pos[0].detach().cpu()) * box_size],
                              1)
                self.point_layer.data = torch.stack(
                    [p[:, 0], p[:, 3], p[:, 2], p[:, 1]], 1)

        def onSelect(self, line):
            path = torch.tensor(np.array(line))
            xval, yval = bezier_curve(path, nTimes=200)
            path = torch.tensor(np.stack([xval, yval], 1))
            n_points, new_points = make_equidistant(xval, yval, 100)
            path = torch.tensor(new_points)
            mu = path[0:2].float()
            vols = []
            poss = []
            ppath = []
            if self.latent.shape[1] > 2:
                for j in range(path.shape[0]):
                    dist = torch.linalg.norm(torch.tensor(
                        self.z) - path[j].unsqueeze(0), dim=1)
                    inst_ind = torch.argmin(dist, 0)
                    ppath.append(self.latent[inst_ind])
                path = torch.stack(ppath, 0)
                path = torch.unique(path, dim=0)

            if self.rep_menu.current_choice == 'volume':
                print('Generating movie with', path.shape[0], 'frames')
                for i in tqdm(range(path.shape[0] // 2)):
                    mu = path[i:i + 2].float()
                    with torch.no_grad():
                        V = self.decoder.volume([mu.to(device)], self.r.to(device),
                                                self.cons_model.pos.to(device),
                                                self.cons_model.amp.to(device),
                                                self.cons_model.ampvar.to(device), self.t.to(device))
                        if atomic_model != None:
                            proj, proj_im, proj_pos, pos, dis = self.decoder.forward([mu.to(device)],
                                                                                     self.r.to(
                                device),
                                pdb_pos.to(
                                device),
                                torch.ones(
                                pdb_pos.shape[
                                    0]).to(
                                device),
                                torch.ones(
                                n_classes,
                                pdb_pos.shape[
                                    0]).to(
                                device),
                                self.t.to(device))
                            # c_pos = inv_half1([mu.to(device)],pos)
                            # points2pdb(args.pdb_series,args.out_dir+'/backwardstate'+ str(i).zfill(3)+'.pdb',c_pos[0]*ang_pix*box_size)
                            # points2pdb(args.pdb_series,args.out_dir+'/forward_state'+ str(i).zfill(3)+'.cif',pos[0]*ang_pix*box_size)
                        # vols.append((V[0]/torch.max(V[0])).float().cpu())
                        # vols.append((V[1]/torch.max(V[1])).float().cpu())
                        vols.append((V[0]).float().cpu())
                        vols.append((V[1]).float().cpu())
                VV = torch.stack(vols, 0)
                VV = VV / torch.max(VV)
                self.vol_layer.data = VV.numpy()
            if self.rep_menu.current_choice == 'points':
                for i in range(path.shape[0] // 2):
                    mu = path[i:i + 2].float()
                    print(i)
                    with torch.no_grad():
                        proj, proj_im, proj_pos, pos, dis = self.decoder.forward([mu.to(device)],
                                                                                 self.r.to(
                            device),
                            self.cons_model.pos.to(
                            device),
                            self.cons_model.amp.to(
                            device),
                            self.cons_model.ampvar.to(
                            device),
                            self.t.to(device))
                        poss.append(torch.cat([i * torch.ones(pos.shape[1]).unsqueeze(1).to(device),
                                               (0.5 + pos[0]) * box_size], 1))
                        poss.append(torch.cat(
                            [(i + 1) * torch.ones(pos.shape[1]).unsqueeze(1).to(device),
                             (0.5 + pos[1]) * box_size], 1))

                PP = torch.cat(poss, 0)
                self.point_layer.data = PP.cpu().numpy()

        def get_part_nr(self, verts):
            path = Path(verts)
            new_indices = path.contains_points(self.z)
            num = np.sum(new_indices)
            print('Number of particles in the selected area is:', num)
            self.poly.disconnect_events()
            self.poly = PolygonSelector(ax=self.axes1, onselect=self.get_part_nr, props=self.line,
                                        useblit=True)

        def save_starfile(self, verts):
            path = Path(verts)
            new_indices = path.contains_points(self.z)
            print(self.z[new_indices])
            new_star = self.star.copy()
            ninds = np.where(new_indices == True)
            print(ninds[0])
            nindsfull = indices[ninds[0]]
            # print(nindsfull)
            print(np.max(self.z[new_indices]))
            print(np.min(self.z[new_indices]))
            new_star['particles'] = self.star['particles'].loc[list(nindsfull)]
            print('created star file with ', len(ninds[0]), 'particles')
            starfile.write(new_star, output_directory + 'subset_' + str(self.star_nr) + '_half' + str(
                half_set) + '.star')
            np.savez(
                output_directory + 'subset_' +
                str(self.star_nr) + '_indices_' + str(half_set) + '.npz',
                nindsfull)
            self.star_nr += 1
            self.poly.disconnect_events()
            self.poly = PolygonSelector(ax=self.axes1, onselect=self.save_starfile, props=self.line,
                                        useblit=True)

    vis = Visualizer(zz, z_lat, decoder, cons_model, V0, cons_volume, nap_cons_pos, dd, amps,
                     lat_colors, dataframe, indices, latent_closest)

    vis.run()
