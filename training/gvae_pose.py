#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:31:48 2021

@author: schwab
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import mrcfile

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
from ..data.handlers.io_logger import IOLogger
from ..utils.utils_new import *
from ..models import *
from coarse_grain import *
# TODO: add coarse graining to GitHub



import warnings

warnings.simplefilter('error', UserWarning)

parser = argparse.ArgumentParser(description='Flexibility VAE')

parser.add_argument('input', help='input job (job directory or optimizer-file)', type=str)
parser.add_argument('log_dir', type=str, metavar='log_dir', help='path to save model checkpoints and intermediate outputs')
parser.add_argument('--ini_model', type=str, help='initial model from prior reconstruction. If not provided the locations of the gaussians are randomly placed in the center of the box. If not provided, the number of consensus epochs (--cons) must be higher')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for Adam optimization')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--gpu', dest='gpu', type=str, default="-1", help='gpu to use')
parser.add_argument('--random_seed', type=int, default='98426')
parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=int(200), help='Total number of training epochs')
parser.add_argument('--pytorch_threads', type=int, default=8)
parser.add_argument('--preload', action='store_true')
parser.add_argument('--dataloader_threads', type=int, default='8')
parser.add_argument('--latent_dim', type=int, default=2, help='Dimensionality of latent space')
parser.add_argument('--n_residues', type=int, default=None, help='Estimated number of residues to choose the number of gaussians')
parser.add_argument('--n_gauss', type=int, default=20000, help='Number of gaussian basis functions')
parser.add_argument('--n_layers', type=int, default=5, help='Number of linear layers for the deformation model')
parser.add_argument('--n_neurons', type=int, default=32, help='Number of neurons in each layer in the deformation model')
parser.add_argument('--cons', type=int, default=None, help='Number of epochs for initial consensus reconstruction in the beginning of training. Not needed if initialization is provided')
parser.add_argument('--particle_diameter', help='size of circular mask (ang)', type=int, default=None)
parser.add_argument('--circular_mask_thickness', help='thickness of mask (ang)', type=int, default=20)
parser.add_argument('--symmetry', help='Soft symmetry penalty (dev)', type=int, default=1)
parser.add_argument('--n_widths', help='number of different gaussians (for determining location of sparse areas of lower resolution)', type=int, default=1)
parser.add_argument('--pos_enc_dim', help='dimension parameter for positional encoding', type=int, default=10)
parser.add_argument('--mask', help='directory of mask file (needs changes in the file)', type= str)
parser.add_argument('--substitute', help = 'update consensus model by taking a step of subsitute value in direction of closest decoder output', type = float, default = None)
parser.add_argument('--random_half', help='Use two random halves for learning the gaussian displacements',action='store_true')
parser.add_argument('--checkpoint', help = 'Start network training from pretrained network', type = str)
parser.add_argument('--model', help = 'atomic model for regularization of deformation if available', action= 'append', default = None)
parser.add_argument('--calibrate_loss_weights', help = 'if true prior-data weighting is applied', default = True)
parser.add_argument('--regularization', help = 'weighting between regularization term and data term (1 = equal weighting, 0 = no regularization), more rigidity for high value',type = float,default = 0.2)
parser.add_argument('--add_free_gaussians', help = 'number of unconstrained gaussians',type = int, default = 0)
parser.add_argument('--bfactor', help = 'flatten power-spectrum of images',type = float, default = 0)

args = parser.parse_args()

batch_size = args.batch_size
log_dir = args.log_dir
summ = SummaryWriter(log_dir)
torch.set_num_threads(args.pytorch_threads)
sys.stdout = IOLogger(os.path.join(log_dir, 'std.out'))

print(args)

if args.gpu == "-1":
    print("Training on available device")
    a = torch.cuda.current_device()
    print(a)
else:
    print("Training on GPU(s)", args.gpu)

multi_gpu = args.gpu.find(",") != -1  # Do we have a comma separated list?
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

device = 'cuda:' + str(a) if args.gpu == "-1" else 'cuda:' + str(args.gpu)
print(device)

print('Initializing the dataset')

dataset,diameter_ang,box_size,ang_pix,optics_group = initialize_dataset(args.input, args.circular_mask_thickness,args.preload,args.particle_diameter)

'----------------------------------------------------------------'
'Define standard oversampling factor'
'----------------------------------------------------------------'

if box_size<100:
    oversampling = 3
if box_size<300 and box_size>99:
    oversampling = 2
if box_size> 299:
    oversampling = 1
    
print('Number of particles:', len(dataset))

'----------------------------------------------------------------------------------------------------------------------------------'
'Initialize poses and pose optimizer'
'----------------------------------------------------------------------------------------------------------------------------------'
original_angles = dataset.part_rotation.astype(np.float32)
original_shifts = -dataset.part_translation.astype(np.float32)
angles = original_angles
shifts = original_shifts  

angles = torch.nn.Parameter(torch.tensor(angles, requires_grad=True).to(device))
angles_op = torch.optim.Adam([angles], lr=1e-3)
shifts = torch.nn.Parameter(torch.tensor(shifts, requires_grad=True).to(device))
shifts_op = torch.optim.Adam([shifts], lr=1e-3)

'--------------------------------------------------------------------------------------------'
'Initialize training dataloaders for random halves and regularization parameter calibration'
'--------------------------------------------------------------------------------------------'

if args.calibrate_loss_weights:
    calibration_data = torch.utils.data.Subset(dataset,torch.randint(0,len(dataset),(100,)))
    data_loader_cal = DataLoader(
        dataset=calibration_data,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=False
        )
        
if args.random_half:
    if args.checkpoint:
        cp = torch.load(args.checkpoint,map_location=device)
        inds_half1 = cp['indices_half1'].cpu().numpy()
        inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))
        print(len(inds_half1))
        print(len(inds_half2))
        dataset_half1 = torch.utils.data.Subset(dataset,inds_half1)
        dataset_half2 = torch.utils.data.Subset(dataset,inds_half2)
    else:
        dataset_half1, dataset_half2 = torch.utils.data.dataset.random_split(dataset,[len(dataset)//2,len(dataset)-len(dataset)//2])
                                            
    data_loader_half1 = DataLoader(
        dataset=dataset_half1,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
        )

    data_loader_half2 = DataLoader(
        dataset=dataset_half2,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
        )
    
    batch = next(iter(data_loader_half1))
    data_preprocessor = ParticleImagePreprocessor()
    data_preprocessor.initialize_from_stack(
        stack=batch["image"],
        circular_mask_radius=diameter_ang / (2 * optics_group['pixel_size']),
        circular_mask_thickness=args.circular_mask_thickness / optics_group['pixel_size']
    )
    data_preprocessor.set_device(device)
    print('Initialized data loaders for half sets of size', len(dataset_half1),' and ', len(dataset_half2))


else:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_threads,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    batch = next(iter(data_loader))
    data_preprocessor = ParticleImagePreprocessor()
    data_preprocessor.initialize_from_stack(
        stack=batch["image"],
        circular_mask_radius=diameter_ang / (2 * optics_group['pixel_size']),
        circular_mask_thickness=args.circular_mask_thickness / optics_group['pixel_size']
    )
    data_preprocessor.set_device(device)

print('box size:', box_size, 'pixel_size:', ang_pix, 'virtual pixel_size:', 1 / (box_size + 1), ' dimension of latent space: ', args.latent_dim )
latent_dim = args.latent_dim

'--------------------------------------------------------------------'
'Iniitialize gaussian model if pdb is provided'
'--------------------------------------------------------------------'

if args.model:
    mode = 'model'
    cons_model,gr = optimize_coarsegraining(args.model, box_size, ang_pix, device,args.log_dir, args.n_widths,args.add_free_gaussians,resolution = 8)
    cons_model.amp.requires_grad = True
    if args.substitute == None:
        args.substitute = 0

if args.n_residues:
    N_points = args.n_residues
elif args.n_gauss:
    N_points = args.n_gauss

if args.model:
    N_points = cons_model.pos.shape[0]
    args.n_gauss = N_points
    print('Number of used gaussians:', args.n_gauss)

'------------------------------------------------------------------'
'Define learning parameters'
'------------------------------------------------------------------'

N_epochs = args.max_epochs
n_layers = args.n_layers
n_neurons = args.n_neurons
consensus = args.cons
n_classes = args.n_widths
pos_enc_dim = args.pos_enc_dim
learn_consensus = False
LR = 0.001
posLR = 0.001
#LR = 0.0005
#posLR = 0.0005


print('Number of used gaussians:', N_points)

try:
    cons_model.i2F.A = torch.nn.Parameter(torch.linspace(0.01,0.02,n_classes).to(device),requires_grad = False)
    cons_model.i2F.B.requires_grad = False
    cons_model.pos.requires_grad = False
except:
    cons_model = consensus_model(box_size, device, N_points, n_classes,3).to(device)
if args.random_half:
    deformation_half1 = displacement_decoder(box_size, device, latent_dim, N_points, n_classes, n_layers, n_neurons,
                                         lin_block, pos_enc_dim,3).to(device)
    deformation_half2 = displacement_decoder(box_size, device, latent_dim, N_points, n_classes, n_layers, n_neurons,
                                         lin_block, pos_enc_dim,3).to(device)
    encoder_half1 = het_encoder(box_size,latent_dim,1).to(device)
    encoder_half2 = het_encoder(box_size,latent_dim,1).to(device)
else:    
    deformation = displacement_decoder(box_size, device, latent_dim, N_points, n_classes, n_layers, n_neurons,
                                     lin_block, pos_enc_dim,1).to(device)
    encoder = het_encoder(box_size,latent_dim).to(device)

if args.ini_model:
    mode = 'density'
    with mrcfile.open(args.ini_model) as mrc:
        Ivol = torch.tensor(mrc.data)
        if Ivol.shape[0]>360:
            Ivols = torch.nn.functional.avg_pool3d(Ivol[None,None],(2,2,2))
            Ivols = Ivols[0,0]
            th = compute_threshold(Ivol)
            print('Setting threshold for the initialization to:', th)
            cons_model.initialize_points(Ivols.movedim(0, 2).movedim(0, 1), 0.0049)
            initialize_consensus(cons_model,Ivols.to(device),log_dir, N_epochs = 50)
        else:
            th = compute_threshold(Ivol)
            print('Setting threshold for the initialization to:', th)
            cons_model.initialize_points(Ivol.movedim(0, 2).movedim(0, 1), 0.0049)
            initialize_consensus(cons_model,Ivol.to(device),log_dir, N_epochs = 50)




if args.mask:
    with mrcfile.open(args.mask) as mrc:
        mask = torch.tensor(mrc.data)
    mask = mask.movedim(0,2).movedim(0,1)

print('consensus model  initialization finished')



if args.checkpoint:
    encoder_half1, encoder_half2, deformation_half1, deformation_half2 = load_models(args.checkpoint,device,box_size,n_classes)


if args.random_half:
    dec_half1_params = deformation_half1.parameters()
    dec_half2_params = deformation_half2.parameters()
    enc_half1_params = encoder_half1.parameters()
    enc_half2_params = encoder_half2.parameters()
else:
    dec_params = deformation.parameters()
    enc_params = encoder.parameters()

cons_params = cons_model.parameters()

if args.random_half:
    dec_half1_params = add_weight_decay(deformation_half1,weight_decay = 1e-1)
    dec_half2_params = add_weight_decay(deformation_half2,weight_decay = 1e-1)
else:
    dec_params = add_weight_decay(deformation,weight_decay = 1e-4)


if args.random_half:
    dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
    dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)   
    enc_half1_optimizer = torch.optim.Adam(enc_half1_params, lr = LR)
    enc_half2_optimizer = torch.optim.Adam(enc_half2_params, lr = LR)
else:
    dec_optimizer = torch.optim.Adam(dec_params, lr = LR)
    enc_optimizer = torch.optim.Adam(enc_params, lr = LR)
#cons_optimizer = torch.optim.Adam(cons_params, lr=posLR)
cons_optimizer = torch.optim.Adam([cons_model.ampvar], lr=posLR)

print(cons_model.amp.requires_grad)
pixel_size = 1 / (box_size + 1)
min_dist_const = 1 / ang_pix
min_dist = 3 * pixel_size

min_dist = 10

mean_dist = torch.zeros(N_points)
mean_dist_half1 = torch.zeros(N_points)
mean_dist_half2 = torch.zeros(N_points)
ind = torch.ones(cons_model.n_points).bool()


xx = torch.tensor(np.linspace(-1,1,box_size,endpoint = False))
XX,YY = torch.meshgrid(xx,xx)

BF = torch.fft.fftshift(torch.exp(-(args.bfactor*(XX**2+YY**2))), dim = [-1,-2]).to(device)


'--------------------------------------------------------------------------------------------------------------------'
'Start Training'
'--------------------------------------------------------------------------------------------------------------------'

deg = torch.zeros(N_points)
kld_weight = batch_size / len(dataset)
beta = kld_weight*0.0006
#beta = kld_weight*0.0002
#beta = kld_weight*0.00002
gamma = 1
delta = 0
min_dd = 100
distance = 0
epoch_t = 0
if consensus == None:
    consensus = 0

loss_history = []
cons_step = 1
old_loss = 1e8
old_loss_half1 = 1e8
old_loss_half2 = 1e8
FRC_im = torch.ones(box_size,box_size)

with torch.no_grad():
    if args.model:
        pos = cons_model.pos*box_size*ang_pix
        gr1 = gr
        gr2 = gr
        cons_dis = torch.pow(torch.sum((pos[gr1[0]]-pos[gr1[1]])**2,1),0.5)
        distance = 0
        deformation_half1.i2F.B.requires_grad = False
        deformation_half2.i2F.B.requires_grad = False
        cons_model.i2F.B.requires_grad = False
        print(cons_dis)
        print(torch.min(cons_dis))
    else:  
        pos = cons_model.pos*box_size*ang_pix
        cons_positions = pos
        grn = knn_graph(pos,2,num_workers = 8)
        mean_neighbour_dist = torch.mean(torch.pow(torch.sum((pos[grn[0]]-pos[grn[1]])**2,1),0.5))
        print('mean distance in graph:', mean_neighbour_dist, ';This distance is also used to construct the initial graph ')
        distance = mean_neighbour_dist
        gr = radius_graph(pos,distance+distance/2,num_workers = 8)
        gr1 = radius_graph(cons_model.pos*ang_pix*box_size,distance+distance/2,num_workers = 8)
        gr2 =  knn_graph(cons_model.pos,1,num_workers = 8)
        cons_dis = torch.pow(torch.sum((cons_positions[gr1[0]]-cons_positions[gr1[1]])**2,1),0.5)


tot_latent_dim = encoder_half1.latent_dim

half1_indices = []
K = 0
print(deformation_half1.device)

if args.random_half:
    with torch.no_grad():
        print('Computing half-set indices')
        for batch_ndx, sample in enumerate(data_loader_half1):
            idx = sample['idx']
            half1_indices.append(idx)
            if batch_ndx%batch_size==0:
                print('Computing indices', batch_ndx/batch_size, 'of' , int(np.ceil(len(data_loader_half1)/batch_size)))
                
            
        half1_indices = torch.tensor([item for sublist in half1_indices for item in sublist])
        cols = torch.ones(len(dataset))
        cols[half1_indices] = 2
        cons_params = cons_model.parameters()
        cons_optimizer = torch.optim.Adam(cons_params, lr=posLR)
    for epoch in range(N_epochs):
        mean_positions = torch.zeros_like(cons_model.pos,requires_grad = False)
        if args.model == None:
            cons_params = cons_model.parameters()
            cons_optimizer = torch.optim.Adam(cons_params, lr=posLR)
        # if epoch > consensus and (epoch+consensus)%5 == 0 and cons_model.n_points < args.n_gauss:
        #     cons_model.double_points(ang_pix,mean_neighbour_dist)
        #     cons_params = cons_model.parameters()
        #     cons_optimizer = torch.optim.Adam(cons_params, lr=posLR)

        if epoch>-1:
            with torch.no_grad():
                if args.model == None:
                    gr1 = radius_graph(cons_model.pos*ang_pix*box_size,distance+distance/2,num_workers = 8)
                    gr2 =  knn_graph(cons_model.pos,2,num_workers = 8)
                    pos = cons_model.pos*box_size*ang_pix
                    cons_dis = torch.pow(torch.sum((pos[gr1[0]]-pos[gr1[1]])**2,1),0.5)
                else:
                    gr1 = gr
                    gr2 = gr
                    pos = cons_model.pos*box_size*ang_pix
                    cons_dis = torch.pow(torch.sum((pos[gr1[0]]-pos[gr1[1]])**2,1),0.5)
                    print(cons_dis)
                    print(torch.min(cons_dis))
        if epoch < consensus:
            cons_model.pos.requires_grad = False
            N_graph = gr1.shape[1]
        else:
            N_graph = 0
        old_amp = cons_model.amp.detach()
        try:
            gr_diff = gr1.shape[1]-gr_old.shape[1]
            if gr_diff<0:
                print(torch.abs(gr_diff),'Gaussians removed to the neighbour graph')
            else:
                print(torch.abs(gr_diff),'Gaussians added to the neighbour graph')
        except:
            print('no graphs assigned yet')
        angles_op.zero_grad()
        shifts_op.zero_grad()
        min_dd = 100
        av_ind = 0
        new_minpos = torch.zeros_like(cons_model.pos)
        if epoch > 0:
            print('Epoch:', epoch, 'Epoch time:', epoch_t)


        if epoch == consensus and args.model != None:
            deformation_half1.i2F.A = torch.nn.Parameter(cons_model.i2F.A.to(device),requires_grad = True)
            deformation_half1.i2F.B = torch.nn.Parameter(cons_model.i2F.B.to(device),requires_grad = True)
            deformation_half2.i2F.A = torch.nn.Parameter(cons_model.i2F.A.to(device),requires_grad = True)
            deformation_half2.i2F.B = torch.nn.Parameter(cons_model.i2F.B.to(device),requires_grad = True)
            dec_half1_params = deformation_half1.parameters()
            dec_half2_params = deformation_half2.parameters()
            print('pdb model was provided')
            dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
            dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)
            cons_params = cons_model.parameters()
            cons_optimizer = torch.optim.Adam(cons_params, lr=LR)
        elif epoch == consensus and args.model == None:
            deformation_half1.i2F.A = torch.nn.Parameter(cons_model.i2F.A.to(device),requires_grad = True)
            deformation_half1.i2F.B = torch.nn.Parameter(cons_model.i2F.B.to(device),requires_grad = True)
            deformation_half2.i2F.A = torch.nn.Parameter(cons_model.i2F.A.to(device),requires_grad = True)
            deformation_half2.i2F.B = torch.nn.Parameter(cons_model.i2F.B.to(device),requires_grad = True)
            dec_half1_params = deformation_half1.parameters()
            dec_half2_params = deformation_half2.parameters()
            dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
            dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)
            cons_params = cons_model.parameters()
            cons_optimizer = torch.optim.Adam(cons_params, lr=posLR)
            
        running_recloss_half1 = 0
        running_latloss_half1  = 0
        running_total_loss_half1  = 0
        sym_loss_total_half1  = 0
        var_total_loss_half1  = 0
        def_total_loss_half1  = 0
        

        running_recloss_half2  = 0
        running_latloss_half2 = 0
        running_total_loss_half2 = 0
        sym_loss_total_half2 = 0
        var_total_loss_half2 = 0
        def_total_loss_half2 = 0

        
        start_t = time.time()

        latent_space = np.zeros([len(dataset), tot_latent_dim])
        diff = np.zeros([len(dataset),1])

        step = 0
        mean_dist = torch.zeros(cons_model.n_points)
        mean_dist_half1 = torch.zeros(cons_model.n_points)
        mean_dist_half2 = torch.zeros(cons_model.n_points)
        displacement_variance_h1 = torch.zeros_like(mean_dist)
        displacement_variance_h2 = torch.zeros_like(mean_dist)
        
        
        if args.calibrate_loss_weights:
            calibration_data = torch.utils.data.Subset(dataset,torch.randint(0,len(dataset),(1000,)))
            data_loader_cal = DataLoader(
                dataset=calibration_data,
                batch_size=batch_size,
                num_workers=8,
                shuffle=True,
                pin_memory=False
                )
            enc_norm_tot = 0
            dec_norm_tot = 0
            print('calibrating loss parameters and data profile')
            for batch_ndx, sample in enumerate(data_loader_cal):
                deformation_half1.zero_grad()
                encoder_half1.zero_grad()
                r, y, ctf, shift = sample["rotation"], sample["image"], sample["ctf"], -sample['translation']
                idx = sample['idx']
                shift_in = shift.to(device)
                r = angles[idx]
                shift = shifts[idx]

                y, r, ctf, shift = y.to(device), r.to(device), ctf.to(device), shift.to(device)


                data_preprocessor.set_device(device)
                y_in = data_preprocessor.apply_square_mask(y)
                y_in = data_preprocessor.apply_translation(y_in.detach(), -shift[:,0].detach(),-shift[:,1].detach())
                y_in = data_preprocessor.apply_circular_mask(y_in)
                #y_in = y
                
                ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

                mu, logsigma = encoder_half1(y_in,ctf)
                z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
                z_in = [z]


                if epoch < consensus and learn_consensus == True:  # Set latent code for consensus reconstruction to zero
                    Proj, P, PP, n_points = cons_model(r,shift)
                    d_points = torch.zeros_like(n_points)
                else:
                    Proj, P, PP, n_points, d_points = deformation_half1(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                                  cons_model.ampvar.to(device),shift.to(device))
                

                
                
                y = sample["image"].to(device)
                y = data_preprocessor.apply_circular_mask(y.detach())
                rec_loss = Fourier_loss(Proj.squeeze(), y.squeeze(), ctf.float(), W = BF[None,:,:])
                
                rec_loss.backward()
                with torch.no_grad():
                    try:
                        total_norm = 0
    
                        for p in deformation_half1.parameters():
                            if p.requires_grad == True:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                        dec_norm_tot += total_norm ** 0.5
                    except:
                        dec_norm_tot = 1
            
            data_norm = dec_norm_tot
            enc_norm_tot = 0
            dec_norm_tot = 0
            
            with torch.no_grad():
                x = Proj
                N = x.shape[-1]
                yd = torch.fft.fft2(y,dim=[-1,-2],norm = 'ortho')
                x = torch.multiply(x,ctf)
                model_spec, m_s = PowerSpec2(x,batch_reduce = 'mean')
                data_spec, d_s = PowerSpec2(yd, batch_reduce = 'mean')
                model_avg, m_a = RadAvg2(x,batch_reduce = 'mean')
                data_avg, d_a = RadAvg2(BF[None,:,:]*yd,batch_reduce = 'mean')
                data_var, d_v = RadAvg2((data_avg-yd)**2,batch_reduce = 'mean')
                model_var, m_v = RadAvg2((model_avg-x)**2,batch_reduce = 'mean')
                err_avg, e_a = RadAvg2((x-BF[None,:,:]*yd)**2,batch_reduce = 'mean')
                err_im = (x-yd)**2
                err_im_real = torch.real(torch.fft.ifft2(err_im))
                
            for batch_ndx, sample in enumerate(data_loader_cal):
                encoder_half1.zero_grad()
                deformation_half1.zero_grad()
                cons_optimizer.zero_grad()
                angles_op.zero_grad()
                shifts_op.zero_grad()
                r, y, ctf, shift = sample["rotation"], sample["image"], sample["ctf"], -sample['translation']
                idx = sample['idx']
                shift_in = shift.to(device)
                r = angles[idx]
                shift = shifts[idx]

                y, r, ctf, shift = y.to(device), r.to(device), ctf.to(device), shift.to(device)


                data_preprocessor.set_device(device)
                y_in = data_preprocessor.apply_square_mask(y)
                y_in = data_preprocessor.apply_translation(y_in.detach(), -shift[:,0].detach(),-shift[:,1].detach())
                y_in = data_preprocessor.apply_circular_mask(y_in)
                #y_in = y
                
                ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

                mu, logsigma = encoder_half1(y_in,ctf)
                z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
                z_in = [z]


                if epoch < consensus and learn_consensus == True:  # Set latent code for consensus reconstruction to zero
                    Proj, P, PP, n_points = cons_model(r,shift)
                    d_points = torch.zeros_like(n_points)
                else:
                    Proj, P, PP, n_points, d_points = deformation_half1(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                                  cons_model.ampvar.to(device),shift.to(device))
                
                y = sample["image"].to(device)
                y = data_preprocessor.apply_circular_mask(y.detach())
                geo_loss = geometric_loss(n_points, box_size, ang_pix, distance, deformation = cons_dis,graph1 = gr1,graph2 = gr2,mode = mode)
                try:
                    geo_loss.backward()
                except:
                    pass
                with torch.no_grad():
                    try:
                        total_norm = 0
        
                        for p in deformation_half1.parameters():
                            if p.requires_grad == True:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                            dec_norm_tot += total_norm ** 0.5
                    except: dec_norm_tot = 1
                prior_norm = dec_norm_tot
            if epoch == 0:
                lam_reg = 0
            else:
                lam_reg = args.regularization*0.5*data_norm/prior_norm
            #lam_reg = 0
            print('new regularization parameter:', lam_reg)

        

        for batch_ndx, sample in enumerate(data_loader_half1):
            if batch_ndx%100==0:
                print('Processing batch', batch_ndx/batch_size, 'of' , int(np.ceil(len(data_loader_half1)/batch_size)), ' from half 1')
                
            enc_half1_optimizer.zero_grad()
            dec_half1_optimizer.zero_grad()
            cons_optimizer.zero_grad()

            
            r, y, ctf, shift = sample["rotation"], sample["image"], sample["ctf"], -sample['translation']
            idx = sample['idx']
            shift_in = shift.to(device)
            r = angles[idx]
            shift = shifts[idx]

            y, r, ctf, shift = y.to(device), r.to(device), ctf.to(device), shift.to(device)


            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in.detach(), -shift[:,0].detach(),-shift[:,1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)
            #y_in = y
            
            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

            mu, logsigma = encoder_half1(y_in,ctf)
            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
            z_in = [z]


            if epoch < consensus and learn_consensus == True:  # Set latent code for consensus reconstruction to zero
                # Proj, P, PP, n_points = cons_model(r,shift)
                # d_points = torch.zeros_like(n_points)
                Proj, P, PP, n_points, d_points = deformation_half1(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                              cons_model.ampvar.to(device),shift.to(device))
            else:
                Proj, P, PP, n_points, d_points = deformation_half1(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                              cons_model.ampvar.to(device),shift.to(device))
                
                with torch.no_grad():
                    try:
                        frc_half1 += FRC(Proj,y,ctf)
                    except:
                        frc_half1 = FRC(Proj,y,ctf)
                    mu2, logsigma2 = encoder_half2(y_in,ctf)
                    z_in2 = [mu2]
                    _, _, _, _, d_points2 = deformation_half2(z_in2, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                                  cons_model.ampvar.to(device),shift.to(device))
                    mean_positions += torch.sum(n_points.detach(),0)

                    diff[sample["idx"].cpu().numpy()] = torch.mean(torch.sum((d_points-d_points2)**2,2),1).unsqueeze(1).detach().cpu()
                    
                
            displacement_variance_h1 += torch.sum(torch.linalg.norm(d_points.detach().cpu(),dim = 2)**2,0)
            y = sample["image"].to(device)
            y = data_preprocessor.apply_circular_mask(y.detach())
            rec_loss = Fourier_loss(Proj.squeeze(), y.squeeze(), ctf.float(), W = BF[None,:,:])
            latent_loss = -0.5 * torch.mean(torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma), dim=1), dim=0)

            st = time.time()
            sym_loss = torch.zeros(1).to(device)

            if epoch < consensus: #and cons_model.n_points<args.n_gauss:
                geo_loss = torch.zeros(1).to(device)

            else:
                encoder_half1.requires_grad = True
                geo_loss = geometric_loss(n_points, box_size, ang_pix, distance,deformation = cons_dis, graph1 = gr1,graph2 = gr2,mode = mode)


            if epoch < consensus:
                loss = rec_loss + beta *kld_weight * latent_loss
            else:
                loss = rec_loss + beta *kld_weight * latent_loss  + lam_reg*geo_loss 
            
            loss.backward()
            if epoch < consensus:
                cons_optimizer.step()

            else:
                encoder_half1.requires_grad = True
                deformation_half1.requires_grad = True
                deformation_half2.requires_grad = True
                enc_half1_optimizer.step()
                dec_half1_optimizer.step()
                cons_optimizer.step()


            
            eval_t = time.time() - st
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            running_recloss_half1 += rec_loss.item()
            running_latloss_half1 += latent_loss.item()
            running_total_loss_half1 += loss.item()
            var_total_loss_half1 += geo_loss.item()


        for batch_ndx, sample in enumerate(data_loader_half2):
            if batch_ndx%100==0:
                print('Processing batch', batch_ndx/batch_size, 'of', int(np.ceil(len(data_loader_half2)/batch_size)), 'from half 2')
            enc_half2_optimizer.zero_grad()
            dec_half2_optimizer.zero_grad()
            cons_optimizer.zero_grad()


            r, y, ctf, shift = sample["rotation"], sample["image"], sample["ctf"], -sample['translation']
            idx = sample['idx']
            shift_in = shift.to(device)
            r = angles[idx]
            shift = shifts[idx]
            y, r, ctf, shift = y.to(device), r.to(device), ctf.to(device), shift.to(device)


            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in.detach(), -shift[:,0].detach(),-shift[:,1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)
            #y_in = y
            
            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

            mu, logsigma = encoder_half2(y_in,ctf)

            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)

            z_in = [z]


            if epoch < consensus and learn_consensus == True:  # Set latent code for consensus reconstruction to zero
                # Proj, P, PP, n_points = cons_model(r,shift)
                # d_points = torch.zeros_like(n_points)
                # d_points2 = torch.zeros_like(n_points)
                Proj, P, PP, n_points, d_points = deformation_half2(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                              cons_model.ampvar.to(device),shift.to(device))
            else:
                Proj, P, PP, n_points, d_points = deformation_half2(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                              cons_model.ampvar.to(device),shift.to(device))
                
                with torch.no_grad():
                    try:
                        frc_half2 += FRC(Proj,y,ctf)
                    except:
                        frc_half2 = FRC(Proj,y,ctf)
                    mu2, logsigma2 = encoder_half1(y_in,ctf)
                    z_in2 = [mu2]
                    _, _, _, _, d_points2 = deformation_half1(z_in2, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                                  cons_model.ampvar.to(device),shift.to(device))
                    mean_positions += torch.sum(n_points.detach(),0)

                    diff[sample["idx"].cpu().numpy()] = torch.mean((d_points-d_points2)**2).detach().cpu()
                    
                
            displacement_variance_h2 += torch.sum(torch.linalg.norm(d_points.detach().cpu(),dim = 2)**2,0)
            y = sample["image"].to(device)
            y = data_preprocessor.apply_circular_mask(y.detach())
            rec_loss = Fourier_loss(Proj.squeeze(), y.squeeze(), ctf.float(), W = BF[None,:,:])
            latent_loss = -0.5 * torch.mean(torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma), dim=1), dim=0)

            st = time.time()

            if epoch < consensus: #and cons_model.n_points<args.n_gauss:
                geo_loss = torch.zeros(1).to(device)
            
            else:
                encoder_half2.requires_grad = True
                geo_loss = geometric_loss(n_points, box_size, ang_pix, distance,deformation = cons_dis, graph1 = gr1,graph2 = gr2,mode = mode)

            
            if epoch<consensus:
                loss = rec_loss + beta *kld_weight * latent_loss
            else:
                loss = rec_loss + beta *kld_weight * latent_loss + lam_reg*geo_loss 

            loss.backward()
            if epoch < consensus:
                cons_optimizer.step()

            else:
                encoder_half2.requires_grad = True
                deformation_half1.requires_grad = True
                deformation_half2.requires_grad = True
                enc_half2_optimizer.step()
                dec_half2_optimizer.step()
                cons_optimizer.step()

            
            eval_t = time.time() - st
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            running_recloss_half2 += rec_loss.item()
            running_latloss_half2 += latent_loss.item()
            running_total_loss_half2 += loss.item()
            var_total_loss_half2 += geo_loss.item()
           
            with torch.no_grad():
                mean_dist_half1 += torch.sum(torch.linalg.norm(d_points, dim=2), 0).cpu()
                mean_dist_half2 += torch.sum(torch.linalg.norm(d_points2, dim=2), 0).cpu()

        angles_op.step()
        shifts_op.step()
        
        current_angles = angles.detach().cpu().numpy()
        angular_error = np.mean(np.square(current_angles - original_angles))

        current_shifts = shifts.detach().cpu().numpy()
        translational_error = np.mean(np.square(current_shifts - original_shifts))

        poses = pose_model(box_size, device, torch.tensor(current_angles), torch.tensor(current_shifts))
        try:
            gr_old = gr1
        except:
            print('no graphs available yet')

        if epoch > (consensus-1) and args.substitute != 0:
            min_dd = 10000000
            mean_positions /= len(dataset)
            if K%2 == 0:
                dl = data_loader_half1
            else:
                dl = data_loader_half2
            K += 1
            with torch.no_grad():
                for batch_ndx, sample in enumerate(dl):
                    r, y, ctf, shift = sample["rotation"], sample["image"], sample["ctf"], -sample['translation']
                    idx = sample['idx'].to(device)
                    #shift_in = shift.to(device)
                    r = angles[idx]
                    shift = shifts[idx]
                    y, r, ctf, shift = y.to(device), r.to(device), ctf.to(device), shift.to(device)

                    data_preprocessor.set_device(device)
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in.detach(), -shift[:,0].detach(),-shift[:,1].detach())
                    y_in = data_preprocessor.apply_circular_mask(y_in)
                    #y_in = y
                    
                    ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

                    mu, logsigma = encoder_half1(y_in,ctf)

                    z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
                    z_in = [z]
                    #z_amp = z[:,-1:]
                    Proj, P, PP, n_points, d_points = deformation_half1(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                                      cons_model.ampvar.to(device),shift.to(device))
                    defs = torch.linalg.norm(d_points, dim=2)
                    mean_def_part = torch.mean(defs,1)
                    dd = torch.min(mean_def_part)
                    if dd < min_dd:
                        min_ind = torch.argmin(mean_def_part, dim=0)
                        min_npos = n_points[min_ind]
                        min_dpos = d_points[min_ind]
                        min_dd = dd
            #     #print('Mean deformation over the complete dataset:', torch.mean(mean_def_part))
            #     #print('Smallest deformation from consensus in the dataset:', dd)
                #print('Largest displacement in this minimal deformation:', torch.max(d_points[min_ind]))
            if (running_recloss_half1 < old_loss_half1 and running_recloss_half2 < old_loss_half2) and args.substitute !=0:        
                cons_model.pos = torch.nn.Parameter((1-args.substitute)*cons_model.pos+args.substitute*min_npos,requires_grad=False)
                old_loss_half1 = running_recloss_half1
                old_loss_half2 = running_recloss_half2
                nosub_ind = 0
            if running_recloss_half1 > old_loss_half1 and running_recloss_half2 > old_loss_half2:
                nosub_ind += 1
                print('No consensus updates for ', nosub_ind, ' epochs')
                if nosub_ind ==1:
                    args.substitute *= 0.85
                if args.substitute<0.2: 
                    args.substitute = 0
                    reset_all_linear_layer_weights(deformation_half1)
                    reset_all_linear_layer_weights(deformation_half2)
                    reset_all_linear_layer_weights(encoder_half1)
                    reset_all_linear_layer_weights(encoder_half2)
                    args.regularization = 1
            #if epoch>5:
            #    args.substitute = 0

        with torch.no_grad():
            # if epoch == 30:
            #     args.substitute = 1
            # else:
            #     args.substitute = 0
            frc_half1 /= len(dataset_half1)
            frc_half2 /= len(dataset_half2)
            frc_both = (frc_half1+frc_half2)/2
            frc_both = frc_both[:box_size//2]
            #FRC_im = prof2radim(frc_both)
            #FRC_im[torch.isnan(FRC_im)] = 0
            #FRC_im[torch.abs(FRC_im)>1] = 0
            mean_dist = mean_dist / latent_space.shape[0]
            pos = cons_model.pos*box_size*ang_pix
            grn = knn_graph(pos,2,num_workers = 8)
            mean_neighbour_dist = torch.mean(torch.pow(torch.sum((pos[grn[0]]-pos[grn[1]])**2,1),0.5))
            print('mean distance in graph in Angstrom:', mean_neighbour_dist)
            distance = mean_neighbour_dist
            #distance = 4
            displacement_variance_h1 /= len(dataset_half1) 
            displacement_variance_h2 /= len(dataset_half2) 
            D_var = torch.stack([displacement_variance_h1,displacement_variance_h2],1)
            print(mean_positions[0])
            print(cons_model.pos[0])
            print(deformation_half1.i2F.B, cons_model.i2F.B)
            print(deformation_half1.i2F.A, cons_model.i2F.A)
            if args.model == None:
                gr = radius_graph(pos,distance+distance/2,num_workers = 8)
            graph2bild(pos, gr, log_dir + '/graph' + str(epoch).zfill(3),color = epoch%65)
            ff = generate_form_factor(cons_model.i2F.A, cons_model.i2F.B, box_size)
            ff2 = generate_form_factor(deformation_half1.i2F.A,deformation_half1.i2F.B,box_size)
            ff2b = generate_form_factor(deformation_half2.i2F.A,deformation_half1.i2F.B,box_size)
            FF = np.concatenate([ff,ff2,ff2b],1)
            correct_spec = prof2radim(cons_model.W)
            WW = torch.fft.fftshift(correct_spec,dim = [-2,-1])
            ind1 = torch.randint(0,box_size-1,(1,1))
            ind2 = torch.randint(0,box_size-1,(1,1))
            err_pix = err_im[:,ind1,ind2]
            
            x = Proj[0]
            yd = y[0]
            if x.is_complex():
                pass
            else:
                x = torch.fft.fft2(x,dim=[-2,-1],norm = 'ortho')
            N = x.shape[-1]
            
            
            if tot_latent_dim > 2:
                if epoch % 5 == 0 and epoch > consensus:
                    summ.add_figure(f"Data/latent", visualize_latent(latent_space,c=diff/(np.max(diff)+1e-7) , s=3, alpha=0.2,method = 'pca'),
                                    epoch)
                    summ.add_figure(f"Data/latent2", visualize_latent(latent_space, c = cols, s = 3, alpha = 0.2, method = 'pca'))

            else:
                summ.add_figure(f"Data/latent",
                                visualize_latent(latent_space, c=diff/(np.max(diff)+1e-22), s=3, alpha=0.2),
                                epoch)
                summ.add_figure(f"Data/latent2",
                                visualize_latent(latent_space, c=cols, s=3, alpha=0.2),
                                epoch)
            summ.add_scalar(f"Loss/kld_loss", (running_latloss_half1+running_latloss_half2) / (len(data_loader_half1)+len(data_loader_half2)), epoch)
            summ.add_scalar(f"Loss/mse_loss", (running_recloss_half1+running_recloss_half2) / (len(data_loader_half1)+len(data_loader_half2)), epoch)
            summ.add_scalars(f"Loss/mse_loss_halfs", {'half1': (running_recloss_half1) / (len(data_loader_half1)),'half2': (running_recloss_half2) / (len(data_loader_half2))}, epoch)
            summ.add_scalar(f"Loss/total_loss", (running_total_loss_half1+running_total_loss_half2) / (len(data_loader_half1)+len(data_loader_half2)), epoch)
            summ.add_scalar(f"Loss/dist_loss", (var_total_loss_half1+var_total_loss_half2) / (len(data_loader_half1)+len(data_loader_half2)), epoch)
            if epoch < consensus:
                summ.add_scalar(f"Loss/variance", cons_model.i2F.B[0].detach().cpu(), epoch)
            else:
                summ.add_scalar(f"Loss/variance1a", deformation_half1.i2F.B[0].detach().cpu(), epoch)
                summ.add_scalar(f"Loss/variance2a", deformation_half2.i2F.B[0].detach().cpu(), epoch)
                
            summ.add_figure(f"Data/cons_amp", tensor_plot(cons_model.amp.detach()), epoch)        
            #summ.add_scalar(f"Loss/variance2", deformation.i2F.B[1].detach().cpu(), epoch)
            summ.add_scalar(f"Loss/N_graph", N_graph, epoch)
            summ.add_scalar(f"Loss/substitute", args.substitute, epoch)
            summ.add_scalar(f"Loss/pose_error", angular_error, epoch)
            summ.add_scalar(f"Loss/trans_error", translational_error, epoch)
            summ.add_figure(f"Data/output", tensor_imshow(torch.fft.fftshift(apply_CTF(Proj[0], ctf[0].float()).squeeze().cpu(),dim = [-1,-2])), epoch)
            summ.add_figure(f"Data/input", tensor_imshow(y_in[0].squeeze().detach().cpu()), epoch)
            summ.add_figure(f"Data/target", tensor_imshow(torch.fft.fftshift(apply_CTF(y[0], BF.float()).squeeze().cpu(),dim = [-1,-2])), epoch)
            summ.add_figure(f"Data/cons_points_z_half1",
                            tensor_scatter(cons_model.pos[:, 0], cons_model.pos[:, 1], c=mean_dist_half1, s=3), epoch)
            summ.add_figure(f"Data/cons_points_z_half2",
                            tensor_scatter(cons_model.pos[:, 0], cons_model.pos[:, 1], c=mean_dist_half2, s=3), epoch)
            summ.add_figure(f"Data/delta", tensor_scatter(n_points[0, :, 0], n_points[0, :, 1], 'b', s=0.1), epoch)
            summ.add_figure(f"Data/delta_h1", tensor_scatter(d_points2[0, :, 0],d_points2[0, :, 1], 'b', s=0.1), epoch)
            #summ.add_figure(f"Data/min_n_pos", tensor_scatter(min_npos[:, 0], min_npos[:, 1], 'b', s=0.1), epoch)

            summ.add_figure(f"Data/projection_image",
                            tensor_imshow(torch.fft.fftshift(torch.real(torch.fft.ifftn(Proj[0],dim = [-1,-2])).squeeze().detach().cpu(),dim = [-1,-2])), epoch)
            summ.add_figure(f"Data/mod_model", tensor_imshow(apply_CTF(apply_CTF(y[0], ctf[0].float()),FRC_im.to(device)).squeeze().detach().cpu()), epoch)
            summ.add_figure(f"Data/shapes", tensor_plot(FF), epoch)
            summ.add_figure(f"Data/dis_var", tensor_plot(D_var), epoch)
            summ.add_figure(f"Data/radial_prof",tensor_imshow(WW.cpu()),epoch)
            summ.add_figure(f"Data/model_spec", tensor_plot(m_a), epoch)
            summ.add_figure(f"Data/data_spec", tensor_plot(d_a), epoch)
            summ.add_figure(f"Data/err_prof", tensor_plot(e_a), epoch)
            summ.add_figure(f"Data/avg_err",tensor_imshow(torch.fft.fftshift(err_avg,dim = [-1,-2]).cpu()),epoch)
            summ.add_figure(f"Data/err_im",tensor_imshow(err_im_real[0].cpu()),epoch)
            summ.add_figure(f"Data/err_hist_r",tensor_hist(torch.real(err_pix).cpu(),40),epoch)
            summ.add_figure(f"Data/err_hist_c",tensor_hist(torch.imag(err_pix).cpu(),40),epoch)
            summ.add_figure(f"Data/frc_h1", tensor_plot(frc_half1), epoch)
            summ.add_figure(f"Data/frc_h2", tensor_plot(frc_half2), epoch)
            
            frc_half1 = torch.zeros_like(frc_half1)
            frc_half2 = torch.zeros_like(frc_half2)
            

            



        epoch_t = time.time() - start_t

        if epoch % 1 == 0:
            with torch.no_grad():
                z0 = torch.zeros(2, 2)
                r0 = torch.zeros(2, 3)
                t0 = torch.zeros(2,2)
                # cpu_cons = cons_model.to('cpu')
                # cpu_cons.device = 'cpu'
                # cpu_cons.i2F.device = 'cpu'
                # cpu_cons.i2F.B = torch.nn.Parameter(deformation_half1.i2F.B.to(cpu_cons.device),requires_grad = False)
                # cpu_cons.i2F.A = torch.nn.Parameter(deformation_half1.i2F.A.to(cpu_cons.device),requires_grad = False)
                V = cons_model.volume(r0.to(device),t0.to(device)).cpu()
                # cons_model.to(device)
                # cons_model.device = device
                # cons_model.i2F.device = device
                #cons_model.i2F.B = torch.nn.Parameter(deformation_half1.i2F.B.to(device),requires_grad = True)
                #fsc,res = FSC(V[0].cpu(),Ivol,ang_pix=ang_pix,visualize = True)
                types = torch.argmax(torch.nn.functional.softmax(cons_model.ampvar, dim=0), dim=0)
                checkpoint = {'encoder_half1': encoder_half1,
                              'encoder_half2': encoder_half2,
                              'consensus': cons_model,
                              'decoder_half1': deformation_half1,
                              'decoder_half2': deformation_half2,
                              'poses': poses,
                              'encoder_half1_state_dict': encoder_half1.state_dict(),
                              'encoder_half2_state_dict': encoder_half2.state_dict(),
                              'consensus_state_dict': cons_model.state_dict(),
                              'decoder_half1_state_dict': deformation_half1.state_dict(),
                              'decoder_half2_state_dict': deformation_half2.state_dict(),
                              'poses_state_dict': poses.state_dict(),
                              'enc_half1_optimizer': enc_half1_optimizer.state_dict(),
                              'enc_half2_optimizer': enc_half2_optimizer.state_dict(),
                              'cons_optimizer': cons_optimizer.state_dict(),
                              'dec_half1_optimizer': dec_half1_optimizer.state_dict(),
                              'dec_half2_optimizer': dec_half2_optimizer.state_dict(),
                              'indices_half1': half1_indices}
                torch.save(checkpoint, log_dir + '/checkpoint' + str(epoch).zfill(3) + '.pth')
                points2xyz(cons_model.pos, log_dir + '/positions' + str(epoch).zfill(3), box_size, ang_pix, types)
                points2xyz(mean_positions, log_dir + '/mean_positions' + str(epoch).zfill(3), box_size, ang_pix, types)
                #points2xyz(maskpoints(cons_model.pos,cons_model.ampvar,deformation.mask[0],box_size)[0], log_dir + '/masked_positions' + str(epoch).zfill(3), box_size, ang_pix, types)
                
                with mrcfile.new(log_dir + '/volume' + str(epoch).zfill(3) + '.mrc', overwrite=True) as mrc:
                    #mrc.set_data((V[0]/torch.mean(V[0])).float().detach().cpu().numpy())
                    mrc.set_data((V[0]/torch.mean(V[0])).float().numpy())
                #del V

    
    
else:    
    
    
    for epoch in range(N_epochs):
        if epoch > consensus and (epoch+consensus)%5 == 0 and cons_model.n_points < args.n_gauss:
            cons_model.double_points(ang_pix,mean_neighbour_dist)
            cons_params = cons_model.parameters()
            cons_optimizer = torch.optim.Adam(cons_params, lr=posLR)
            if cons_model.n_points<args.n_gauss and cons_model.n_points > args.n_gauss//9:
                distance = 3.8
            elif cons_model.n_points>args.n_gauss//2:
                distance = 1.6
            else:
                distance = 1.6
        print(distance)
        if epoch>0:
            with torch.no_grad():
                gr1 = radius_graph(cons_model.pos*ang_pix*box_size,distance+distance/2,num_workers = 8)
                gr2 =  knn_graph(cons_model.pos,1,num_workers = 8)
                pos = cons_model.pos*box_size*ang_pix
                cons_dis = torch.pow(torch.sum((pos[gr1[0]]-pos[gr1[1]])**2,1),0.5)
            N_graph = gr1.shape[1]
        else:
            N_graph = 0
        old_amp = cons_model.amp.detach()
        try:
            gr_diff = gr1.shape[1]-gr_old.shape[1]
            if gr_diff<0:
                print(torch.abs(gr_diff),'Gaussians removed to the neighbour graph')
            else:
                print(torch.abs(gr_diff),'Gaussians added to the neighbour graph')
        except:
            print('no graphs assigned yet')
        angles_op.zero_grad()
        shifts_op.zero_grad()
        min_dd = 100
        av_ind = 0
        new_minpos = torch.zeros_like(cons_model.pos)
        if epoch > 0:
            print('Epoch:', epoch, 'Epoch time:', epoch_t)
    
    
        if epoch == consensus:
            deformation.i2F.A = torch.nn.Parameter(cons_model.i2F.A.to(device),requires_grad = False)
            deformation.i2F.B = torch.nn.Parameter(cons_model.i2F.B.to(device),requires_grad = False)
            dec_params = deformation.parameters()
            dec_optimizer = torch.optim.Adam(dec_params, lr=LR)
            cons_params = cons_model.parameters()
            cons_optimizer = torch.optim.Adam(cons_params, lr=posLR)
            
        running_recloss = 0
        running_latloss = 0
        running_total_loss = 0
        sym_loss_total = 0
        var_total_loss = 0
        def_total_loss = 0
        start_t = time.time()
    
        latent_space = np.zeros([len(dataset), tot_latent_dim])
    
        step = 0
        mean_dist = torch.zeros(cons_model.n_points)
    
        for batch_ndx, sample in enumerate(data_loader):
            if batch_ndx%100==0:
                print('Processing batch', batch_ndx)
            
            imizer.zero_grad()
            dec_optimizer.zero_grad()
            cons_optimizer.zero_grad()
    
            r, y, ctf, shift = sample["rotation"], sample["image"], sample["ctf"], -sample['translation']
            idx = sample['idx']
            shift_in = shift.to(device)
            r = angles[idx]
            shift = shifts[idx]
            y, r, ctf, shift = y.to(device), r.to(device), ctf.to(device), shift.to(device)
    
    
            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in.detach(), -shift[:,0].detach(),-shift[:,1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)
    
            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])
    
            mu, logsigma = encoder(y_in,ctf)
    
            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
    
            z_in = [z]
            #z_amp = z[:,-1:]
    
    
            if epoch < consensus and learn_consensus == True:  # Set latent code for consensus reconstruction to zero
                Proj, P, PP, n_points = cons_model(r,shift)
                d_points = torch.zeros_like(n_points)
            else:
                Proj, P, PP, n_points, d_points = deformation(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                              cons_model.ampvar.to(device),shift.to(device))
                
    
            y = sample["image"].to(device)
            y = data_preprocessor.apply_circular_mask(y.detach())
            rec_loss = Fourier_loss(Proj.squeeze(), y.squeeze(), ctf.float(), W = BF[None,:,:])
            latent_loss = -0.5 * torch.mean(torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma), dim=1), dim=0)
    
            st = time.time()
            sym_loss = torch.zeros(1).to(device)
            if cons_model.n_points<args.n_gauss//9:
                encoder.requires_grad = True
                geo_loss = torch.zeros(1).to(device)
            elif epoch < consensus: #and cons_model.n_points<args.n_gauss:
                #def_loss = torch.zeros(1).to(device)
                #var_loss = torch.zeros(1).to(device)
                #encoder.requires_grad = False
                geo_loss = torch.zeros(1).to(device)
                #geo_loss = geometric_loss(cons_model.pos, box_size, ang_pix, distance)
            elif cons_model.n_points>args.n_gauss//15 and cons_model.n_points<args.n_gauss:
                encoder.requires_grad = True
                #var_loss = batch_structure(gr1,gr2,n_points,box_size,ang_pix,device,distance)
                #def_loss = deformation_structure(gr1,n_points,cons_dis,box_size,ang_pix,device)
                #geo_loss = geometric_loss(n_points, box_size, ang_pix, distance,deformation = cons_dis,graph1 = gr1,graph2 = gr2,neighbour = False,distance = True, outlier = True)
                geo_loss = geo_loss = geometric_loss(n_points, box_size, ang_pix, distance,deformation = cons_dis,graph1 = gr1,graph2 = gr2,neighbour = True ,distance = False, outlier = False)
                #geo_loss = torch.zeros(1).to(device)
            else:
                encoder.requires_grad = True
                #var_loss = batch_structure(gr1,gr2,n_points,box_size,ang_pix,device,distance)
                #def_loss = deformation_structure(gr1,n_points,cons_dis,box_size,ang_pix,device)
                geo_loss = geometric_loss(n_points, box_size, ang_pix, distance,deformation = cons_dis, graph1 = gr1,graph2 = gr2,neighbour = True ,distance = True,outlier = False)
                #geo_loss = torch.zeros(1).to(device)
            
    
            loss = rec_loss + beta *kld_weight * latent_loss + 0.0005*np.floor(epoch/100)*geo_loss
    
            loss.backward()
            if epoch < consensus:
                cons_optimizer.step()
    
            else:
                encoder.requires_grad = True
                deformation.requires_grad = True
                enc_optimizer.step()
                dec_optimizer.step()
                cons_optimizer.step()
    
            
            eval_t = time.time() - st
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            running_recloss += rec_loss.item()
            running_latloss += latent_loss.item()
            running_total_loss += loss.item()
            var_total_loss += geo_loss.item()
            #def_total_loss += def_loss.item()
            sym_loss_total += sym_loss.item()
            step += 1
           
            with torch.no_grad():
                av_ind += 1
                min_ind = torch.argmin(torch.mean(torch.linalg.norm(d_points, dim=2), dim=1))
                min_dpos = d_points[min_ind]
                min_npos = n_points[min_ind]
                mean_dist += torch.sum(torch.linalg.norm(d_points, dim=2), 0).cpu()
    
    
        angles_op.step()
        shifts_op.step()
        
        current_angles = angles.detach().cpu().numpy()
        angular_error = np.mean(np.square(current_angles - original_angles))
    
        current_shifts = shifts.detach().cpu().numpy()
        translational_error = np.mean(np.square(current_shifts - original_shifts))
    
        poses = pose_model(box_size, device, torch.tensor(current_angles), torch.tensor(current_shifts))
        try:
            gr_old = gr1
        except:
            print('no graphs available yet')
    
        if epoch > consensus and args.substitute != 0:
            min_dd = 100
            random_indices = torch.randperm(len(dataset))
            data_subset = torch.utils.data.Subset(dataset, random_indices[:10000])
            random_sub_loader = DataLoader(
                dataset=data_subset,
                batch_size=batch_size,
                num_workers=24,
                shuffle=True,
                pin_memory=False
                )
            with torch.no_grad():
                for batch_ndx, sample in enumerate(random_sub_loader):
                    r, y, ctf, shift = sample["rotation"], sample["image"], sample["ctf"], -sample['translation']
                    idx = sample['idx']
                    shift_in = shift
                    r = angles[idx]
                    shift = shifts[idx]
                    y, r, ctf, shift = y.to(device), r.to(device), ctf.to(device), shift.to(device)
    
                    data_preprocessor.set_device(device)
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in.detach(), -shift[:,0].detach(),-shift[:,1].detach())
                    y_in = data_preprocessor.apply_circular_mask(y_in)
    
                    ctf = torch.fft.fftshift(ctf, dim=[-1, -2])
    
                    mu, logsigma = encoder(y_in,ctf)
    
                    z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
                    z_in = [z]
                    #z_amp = z[:,-1:]
                    Proj, P, PP, n_points, d_points = deformation(z_in, r, cons_model.pos.to(device), cons_model.amp.to(device),
                                                                      cons_model.ampvar.to(device),shift.to(device))
                    defs = torch.linalg.norm(d_points, dim=2)
                    mean_def_part = torch.mean(defs,1)
                    dd = torch.min(mean_def_part)
                    if dd < min_dd:
                        min_ind = torch.argmin(mean_def_part, dim=0)
                        min_npos = n_points[min_ind]
                        min_dpos = d_points[min_ind]
                        min_dd = dd
                print('Mean deformation over the complete dataset:', torch.mean(mean_def_part))
                print('Smallest deformation from consensus in the dataset:', dd)
                print('Largest displacement in this minimal deformation:', torch.max(min_dpos))
                if running_recloss < old_loss and args.substitute !=0:        
                    new_pos = min_npos
                    cons_model.pos = torch.nn.Parameter(cons_model.pos+args.substitute*min_dpos,requires_grad=True)
                    #cons_model.pos = torch.nn.Parameter(cons_model.pos+(min_dd/torch.max(min_npos))*min_dpos,requires_grad=True)
                    old_loss = running_recloss
    
                    
            
        with torch.no_grad():
    
            mean_dist = mean_dist / latent_space.shape[0]
            pos = cons_model.pos*box_size*ang_pix
            grn = knn_graph(pos,2,num_workers = 8)
            mean_neighbour_dist = torch.mean(torch.pow((pos[grn[0]]-pos[grn[1]])**2,0.5))
            print('mean distance in graph:', mean_neighbour_dist)
            distance = mean_neighbour_dist
            gr = radius_graph(pos,distance+0.5,num_workers = 8)
            graph2bild(pos, gr, log_dir + '/graph' + str(epoch).zfill(3))
            ff = generate_form_factor(cons_model.i2F.A, cons_model.i2F.B, box_size)
            ff2 = generate_form_factor(deformation.i2F.A,deformation.i2F.B,box_size)
            FF = np.concatenate([ff,ff2],1)
            grn = knn_graph(pos,1,num_workers = 8)
            print(P.shape)
            print(ctf.shape)
            if tot_latent_dim > 2:
                if epoch % 5 == 0 and epoch > consensus:
                    summ.add_figure(f"Data/latent", visualize_latent(latent_space,c=np.linspace(0, 1, latent_space.shape[0]) , s=3, alpha=0.05,method = 'pca'),
                                    epoch)
    
            else:
                summ.add_figure(f"Data/latent",
                                visualize_latent(latent_space, c=cols, s=3, alpha=0.05),
                                epoch)
            summ.add_scalar(f"Loss/kld_loss", running_latloss / len(data_loader), epoch)
            summ.add_scalar(f"Loss/mse_loss", running_recloss / len(data_loader), epoch)
            summ.add_scalar(f"Loss/total_loss", running_total_loss / len(data_loader), epoch)
            summ.add_scalar(f"Loss/dist_loss", var_total_loss / len(data_loader), epoch)
            summ.add_scalar(f"Loss/defo_loss", def_total_loss / len(data_loader), epoch)
            if epoch < consensus:
                summ.add_scalar(f"Loss/variance1", cons_model.i2F.B[0].detach().cpu(), epoch)
            else:
                summ.add_scalar(f"Loss/variance1", deformation.i2F.B[0].detach().cpu(), epoch)
            summ.add_scalar(f"Loss/cons_amp", cons_model.amp.detach().cpu(), epoch)        
            #summ.add_scalar(f"Loss/variance2", deformation.i2F.B[1].detach().cpu(), epoch)
            summ.add_scalar(f"Loss/N_graph", N_graph, epoch)
            summ.add_scalar(f"Loss/pose_error", angular_error, epoch)
            summ.add_scalar(f"Loss/trans_error", translational_error, epoch)
            summ.add_figure(f"Data/output", tensor_imshow(apply_CTF(P[0], ctf[0].float()).squeeze().detach().cpu()), epoch)
            summ.add_figure(f"Data/input", tensor_imshow(y_in[0].squeeze().detach().cpu()), epoch)
            summ.add_figure(f"Data/cons_points_z",
                            tensor_scatter(cons_model.pos[:, 0], cons_model.pos[:, 1], c=mean_dist, s=3), epoch)
            summ.add_figure(f"Data/delta", tensor_scatter(n_points[0, :, 0], n_points[0, :, 1], 'b', s=0.1), epoch)
            #summ.add_figure(f"Data/min_n_pos", tensor_scatter(min_npos[:, 0], min_npos[:, 1], 'b', s=0.1), epoch)
    
            summ.add_figure(f"Data/projection_image",
                            tensor_imshow(apply_CTF(P[0],torch.ones_like(ctf[0])).float().squeeze().detach().cpu()), epoch)
            summ.add_figure(f"Data/shapes", tensor_plot(FF), epoch)
    
    
    
        epoch_t = time.time() - start_t
    
        if epoch % 20 == 0:
            with torch.no_grad():
                z0 = torch.zeros(2, 2)
                r0 = torch.zeros(2, 3)
                t0 = torch.zeros(2,2)
                V = cons_model.volume(r0.to(device),t0.to(device))
                #fsc,res = FSC(V[0].cpu(),Ivol,ang_pix=ang_pix,visualize = True)
                types = torch.argmax(torch.nn.functional.softmax(cons_model.ampvar, dim=0), dim=0)
                checkpoint = {'encoder': encoder,
                              'consensus': cons_model,
                              'decoder': deformation,
                              'poses': poses,
                              'encoder_state_dict': encoder.state_dict(),
                              'consensus_state_dict': cons_model.state_dict(),
                              'decoder_state_dict': deformation.state_dict(),
                              'poses_state_dict': poses.state_dict(),
                              'enc_optimizer': enc_optimizer.state_dict(),
                              'cons_optimizer': cons_optimizer.state_dict(),
                              'dec_optimizer': dec_optimizer.state_dict()}
                torch.save(checkpoint, log_dir + '/checkpoint' + str(epoch).zfill(3) + '.pth')
                points2xyz(cons_model.pos, log_dir + '/positions' + str(epoch).zfill(3), box_size, ang_pix, types)
                #points2xyz(maskpoints(cons_model.pos,cons_model.ampvar,deformation.mask[0],box_size)[0], log_dir + '/masked_positions' + str(epoch).zfill(3), box_size, ang_pix, types)
                
                with mrcfile.new(log_dir + '/volume' + str(epoch).zfill(3) + '.mrc', overwrite=True) as mrc:
                    mrc.set_data((V[0]/torch.mean(V[0])).float().detach().cpu().numpy())
                    mrc.voxel_size = ang_pix
                del V
    
