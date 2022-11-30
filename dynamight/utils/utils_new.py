#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:24:45 2021

@author: schwab
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
# from torch_geometric.nn import radius_graph, knn_graph
# from torch_scatter import scatter
import umap
from sklearn.decomposition import PCA
from tsnecuda import TSNE
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from tqdm import tqdm
import mrcfile
from ..data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
from ..data.dataloaders.relion import RelionDataset
from ..data.handlers.star_file import load_star
from scipy.special import comb

'-----------------------------------------------------------------------------'
'CTF and Loss Functions'
'-----------------------------------------------------------------------------'

def apply_CTF(x,ctf):
    if x.is_complex():
        pass
    else:
        x = torch.fft.fft2(torch.fft.fftshift(x,dim = [-1,-2]),dim=[-1,-2])
    x0 = torch.multiply(x,ctf)
    x0 = torch.fft.ifft2(x0)
    return torch.real(x0)


def Fourier_loss(x,y,ctf,W=None,sig = None):
    if x.is_complex():
        pass
    else:
        x = torch.fft.fft2(torch.fft.fftshift(x,dim = [-1,-2]),dim=[-1,-2],norm = 'ortho')
    y = torch.fft.fft2(torch.fft.fftshift(y,dim = [-1,-2]),dim=[-1,-2],norm = 'ortho')
    x = torch.multiply(x,ctf)
    if W != None:
        y = torch.multiply(y,W)
    # else:
    #     x = torch.multiply(x,ctf)
    l =  torch.mean(torch.pow(torch.mean(torch.abs(x-y)**2,dim=[-1,-2]),0.5))
    return l


def geometric_loss(pos,box_size,ang_pix, dist,mode, deformation = None, graph1 = None, graph2 = None, neighbour = False, distance = False, outlier = False, graph = False):
    pos = pos*box_size*ang_pix
    if mode == 'model':
        graph = True
    if mode == 'density':
        graph = True
        neighbour = True
        distance = True
        
    if len(pos.shape) == 2:
        if neighbour == True:
            gr = radius_graph(pos,dist+0.5,num_workers = 8)
            dis = torch.pow(1e-7+torch.sum((pos[gr[0]]-pos[gr[1]])**2,1),0.5)
            dis2 = distance_activation(dis,dist)
            d = scatter(dis2,gr[0], reduce='sum')
            val = neighbour_activation(d)
            neighbour_loss = torch.mean(val)
        else:
            neighbour_loss = torch.zeros(1).to(pos.device)
        if distance == True:
            try:
                distance_loss = torch.mean(gaussian_distance(dis,dist))
            except:
                gr = radius_graph(pos,distance+0.5,num_workers = 8)
                dis = torch.pow(1e-7+torch.sum((pos[gr[0]]-pos[gr[1]])**2,1),0.5)
                distance_loss = torch.mean(gaussian_distance(dis,dist))
        else:
            distance_loss = torch.zeros(1).to(pos.device)
        if outlier == True:
            gr2 = knn_graph(pos,1,num_workers = 8)
            out_dis = torch.pow(1e-7+torch.sum((pos[gr2[0]]-pos[gr2[1]])**2,1),0.5)
            out_dis = torch.clamp(out_dis,min = dist+0.2)
            outlier_loss = torch.mean(out_dis-(dist+0.2))
        else:
            outlier_loss = torch.zeros(1).to(pos.device)
        deformation_loss = torch.zeros(1).to(pos.device)
    
    elif len(pos.shape) == 3:

        if neighbour == True:
            try:
                graph1
            except:
                print('Radius graph has to be provided for batch application of neighbour and distance loss')
            
            dis = torch.pow(1e-7+torch.sum((pos[:,graph1[0]]-pos[:,graph1[1]])**2,2),0.5)
            dis2 = distance_activation(dis,dist)
            d = scatter(dis2,graph1[0], reduce='sum')
            val = neighbour_activation(d)
            #neighbour_loss = torch.mean(torch.sum(val,1)/pos.shape[1])
            neighbour_loss = torch.mean(torch.mean(val,1))
            #neighbour_loss = torch.mean(torch.sum(val,1))
        else:
            neighbour_loss = torch.zeros(1).to(pos.device)
        if distance == True:
            try:
                dis = torch.pow(1e-7+torch.sum((pos[:,graph1[0]]-pos[:,graph1[1]])**2,2),0.5)
                distance_loss = torch.mean(gaussian_distance(dis,dist))
            except:
                print('Radius graph has to be provided for batch application of neighbour and distance loss')
                distance_loss = torch.zeros(1).to(pos.device)
        else:
            distance_loss = torch.zeros(1).to(pos.device)
        if outlier == True:
            try:
                out_dis = torch.pow(1e-7+torch.sum((pos[:,graph2[0]]-pos[:,graph2[1]])**2,2),0.5)
                out_dis = torch.clamp(out_dis,min = dist+0.2)
                outlier_loss = torch.mean(out_dis-(dist+0.2))
            except:
                print('KNN graph has to be provided for batch application of outlier loss')
        else:
            outlier_loss = torch.zeros(1).to(pos.device)
        if graph == True:
            try:
                dis
            except:
                dis = torch.pow(1e-7+torch.sum((pos[:,graph1[0]]-pos[:,graph1[1]])**2,2),0.5) 
            try:
                diff_dis = torch.abs(dis-deformation)**2
                deformation_loss = torch.mean(diff_dis)
            except:
                print('no distances provided')
        else:
            deformation_loss = torch.zeros(1).to(pos.device)
    
    #print('distance:', distance_loss, 'neighbour:', neighbour_loss, 'outlier:', outlier_loss)

    return 0.0*neighbour_loss + distance_loss + outlier_loss + deformation_loss
        


def gaussian_distance(d,distance):
    if distance<0.5:
        distance = 0.5
    distance = 0.5
    x1 = torch.clamp(d,max = distance)
    x1 = (x1-distance)**2
    #x2 = torch.clamp(d,min = 1.6)
    #x2 = 1-(1-(4.2-2*x2)**2)**2
    return x1

def neighbour_activation(v):
    x1 = torch.clamp(v,max=1)
    x2 = torch.clamp(v,min=3)
    x1 = (x1-1)**2
    #x2 = (1-(4-x2)**2)**2  
    x2 = (x2-3)**2
    return x1+x2

def distance_activation(d,distance):
    # x = torch.zeros_like(d)
    # x1 = torch.clamp(d,max = distance)
    x2 = torch.clamp(d,min = distance,max = distance+0.5*distance)
    #x1 = distance-x1
    #x1 = torch.nn.ReLU(x1)+1
    #x2[x2==distance] = distance + 0.5
    #x[d<distance]=1
    x2 = (1-(4/distance**2)*(x2-distance)**2)**2
    return x2

'-----------------------------------------------------------------------------'
'projection stuff'
'-----------------------------------------------------------------------------'

class point_projection(nn.Module):
   # for angle ordering [TILT,ROT,PSI]
    def __init__(self,box_size):
        super(point_projection,self).__init__()
        self.box_size = box_size

    def forward(self,p,rr):
        device = p.device
        #batch_size = rr.shape[0]
        #yaw = rr[:,1:2]+np.pi
        #pitch = -rr[:,0:1]
        #roll = rr[:,2:3]+np.pi
        if len(rr.shape)<3:
            roll = rr[:,0:1]+np.pi
            yaw = -rr[:,2:3]
            pitch = rr[:,1:2]+np.pi
    
           
            tensor_0 = torch.zeros_like(roll).to(device)
            tensor_1 = torch.ones_like(roll).to(device)
       
            RX = torch.stack([
                torch.stack([torch.cos(roll), -torch.sin(roll), tensor_0]),
                torch.stack([torch.sin(roll), torch.cos(roll), tensor_0]),
                torch.stack([tensor_0, tensor_0, tensor_1])])
       
            RY = torch.stack([
                torch.stack([torch.cos(pitch), tensor_0, -torch.sin(pitch)]),
                torch.stack([tensor_0, tensor_1, tensor_0]),
                torch.stack([torch.sin(pitch), tensor_0, torch.cos(pitch)])])
       
            RZ = torch.stack([
                torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                torch.stack([tensor_0, tensor_0, tensor_1])])
               
            RX = torch.movedim(torch.movedim(RX,3,0),3,0)
            RY = torch.movedim(torch.movedim(RY,3,0),3,0)
            RZ = torch.movedim(torch.movedim(RZ,3,0),3,0)
                    #rotation of P
            Rp = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, -tensor_1, tensor_0]),
                torch.stack([tensor_0, tensor_0, -tensor_1])])
            Rp = torch.movedim(torch.movedim(Rp,3,0),3,0)
            
            
            R = torch.matmul(RZ, RY)
            R = torch.matmul(R, RX)

    
        else:
            R = rr.unsqueeze(1)
            tensor_0 = torch.zeros_like(rr[:,:1,0]).to(device)
            tensor_1 = torch.ones_like(rr[:,:1,0]).to(device)
            Rp = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, -tensor_1, tensor_0]),
                torch.stack([tensor_0, tensor_0, -tensor_1])])
            Rp = torch.movedim(torch.movedim(Rp,3,0),3,0)

        points3 = p#-np.sqrt(2)*1/self.box_size*torch.ones_like(p)
        points2 = torch.matmul(R.squeeze(),torch.matmul(Rp.squeeze(),points3.movedim(1,2)))
        ind = [-3,-2]
        return points2[:,ind,:]


class points2mult_image(nn.Module):
   
    def __init__(self,box_size,n_classes,oversampling =1):
        super(points2mult_image,self).__init__()
        self.box_size = box_size
        self.n_classes = n_classes
        self.os = oversampling


       
    def forward(self, points, values):
        self.batch_size = points.shape[0]
        p = ((points+0.5)*(self.box_size*self.os)).movedim(1,2)
        device = p.device
        im = torch.zeros(self.batch_size,self.n_classes,(self.box_size*self.os)**2).to(device)
        xypoints = p.floor().long()
        rxy = p-xypoints
        x,y = xypoints.split(1,dim = -1)
        rx, ry = rxy.split(1,dim = -1)
       
        for dx in (0,1):
            x_ = x+dx
            wx = (1-dx)+(2*dx-1)*rx
            for dy in (0,1):
                y_ = y+dy
                wy = (1-dy)+(2*dy-1)*ry
                   
                w = wx*wy
                   
                valid = ((0<=x_)*(x_ < self.os*self.box_size)*(0<=y_)*(y_< self.os*self.box_size)).long()
                                       
                idx = (( y_*self.box_size*self.os +x_)*valid).squeeze()
                idx = torch.stack(self.n_classes*[idx],1)
                w = (w*valid.type_as(w)).squeeze()
                w = torch.stack(self.n_classes*[w],1)
                im.scatter_add_(2,idx,w*values)
        im = im.reshape(self.batch_size,self.n_classes,self.os*self.box_size,self.os*self.box_size)

        return im
    
def initialize_consensus(model, ref, logdir,lr = 0.001, N_epochs = 300,mask = None):
    device = model.device
    model_params = model.parameters()
    model_optimizer = torch.optim.Adam(model_params, lr=lr)
    z0 = torch.zeros(2, 2)
    r0 = torch.zeros(2, 3)
    t0 = torch.zeros(2,2)
    print('Initializing gaussian positions from reference deformable_backprojection')
    for i in tqdm(range(N_epochs)):
        model_optimizer.zero_grad()
        V = model.volume(r0.to(device),t0.to(device)).float()
        #fsc,res=FSC(ref,V[0],1,visualize = False)
        loss = torch.nn.functional.mse_loss(V[0],ref)#+1e-7*f1(lay(model.pos))
        loss.backward()
        model_optimizer.step()
    print('Final error:', loss.item())
    with mrcfile.new(logdir + '/ini_volume.mrc', overwrite=True) as mrc:
        mrc.set_data((V[0]/torch.mean(V[0])).float().detach().cpu().numpy())
    



class ims2F_form(nn.Module):
        
    def __init__(self,box_size,device,n_classes,oversampling =1,A = None, B = None):
        super(ims2F_form,self).__init__()
        self.box_size = box_size
        self.device = device
        self.n_classes = n_classes
        self.rad_inds, self.rad_mask = radial_index_mask(oversampling*box_size)
        if A == None and B == None:
            self.B = torch.nn.Parameter(torch.linspace(0.0005*box_size,0.001*box_size,n_classes).to(device),requires_grad = True)
            self.A = torch.nn.Parameter(torch.linspace(0.1,0.2,n_classes).to(device),requires_grad = True)
        else:
            self.B = torch.nn.Parameter(B.to(device),requires_grad = True)
            self.A = torch.nn.Parameter(A.to(device),requires_grad = True)
        self.os = oversampling
        self.crop = fourier_crop
        
    def forward(self, ims):
        R = torch.stack(self.n_classes*[self.rad_inds.to(self.device)],0)
        FF = torch.exp(-self.B[:,None,None]**2*R)*self.A[:,None,None]**2
        bs = ims.shape[0]
        Filts = torch.stack(bs*[FF],0)
        Filts = torch.fft.ifftshift(Filts,dim = [-2,-1])
        Fims = torch.fft.fft2(torch.fft.fftshift(ims,dim = [-2,-1]),norm = 'ortho' )
        if self.n_classes>1:
            out = torch.sum(Filts*Fims,dim = 1)
        else:
            out = Filts*Fims
        if self.os>1:
            out = self.crop(out,self.os)
        return out

def fourier_crop(img,oversampling):
    s = img.shape[-1]
    img = torch.fft.fftshift(img,[-1,-2])
    out = img[...,s//2-s//(2*oversampling):s//2+s//(2*oversampling),s//2-s//(2*oversampling):s//2+s//(2*oversampling)]
    out = torch.fft.fftshift(out,[-1,-2])
    return out

    
def radial_index_mask(box_size,ang_pix=None):
    if ang_pix:
        x = torch.tensor(box_size*ang_pix*np.linspace(-box_size/2,box_size/2,box_size,endpoint = False)/4*np.pi)
    else:
        x = torch.tensor(np.linspace(-box_size,box_size,box_size,endpoint = False))
    X,Y = torch.meshgrid(x,x)
    R = torch.round(torch.sqrt(X**2+Y**2))
    Mask = R<(x[-1])

    return R.long(), Mask

def radial_index_mask3(box_size,ang_pix=None):
    if ang_pix:
        x = torch.tensor(box_size*ang_pix*np.linspace(-box_size/2,box_size/2,box_size,endpoint = False)/4*np.pi)
    else:
        x = torch.tensor(np.linspace(-box_size,box_size,box_size,endpoint = False))
    X,Y,Z = torch.meshgrid(x,x,x)
    R = torch.round(torch.sqrt(X**2+Y**2+Z**2))
    Mask = R<(x[-1])

    return R, Mask

def generate_form_factor(a,b,box_size):
    r = np.linspace(0,box_size,box_size,endpoint = False)
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    R = np.stack(a.shape[0]*[r],0)
    F = np.exp(-b[:,None]**2*R)*a[:,None]**2
    return np.moveaxis(F,0,1)


        
def find_initialization_parameters(model,V):
    r0 = torch.zeros(2, 3).to(model.device)
    t0 = torch.zeros(2,2).to(model.device)
    Vmodel = model.volume(r0,t0)
    ratio = torch.sum(V**2)/torch.sum(Vmodel**2)
    # s = torch.linspace(0,0.5,100)
    # a = torch.linspace(0,2*ratio,100)
    # rad_inds, rad_mask = radial_index_mask3(oversampling*box_size)
    # R = torch.stack(1*[rad_inds.to(model.device)],0)
    # for i in range(100):
    #     for j in range(100):
    #         FF = torch.exp(-s[i,None,None]**2*R)*a[j,None,None]**2
    model.amp = torch.nn.Parameter(0.55*torch.ones(1).to(model.device),requires_grad = True)
   



def initialize_dataset(input_arg,circular_mask_thickness,preload,part_diam=None):
    RD = RelionDataset(input_arg) 
    a = []
    for i in RD.image_file_paths:
        mrc = mrcfile.open(i,'r')
        num_im = mrc.data.shape[0]
        a.append(np.arange(num_im))
    RD.part_stack_idx = np.concatenate(a)
    dataset = RD.make_particle_dataset()
    # dataset = RelionDataset()
    # dataset.project_root = '/beegfs3/kmcnally/Molly2'
    # dataset.data_star_path = input_arg
    # #dataset.load(dataset.data_star_path)
    # data = load_star(dataset.data_star_path)
    # dataset._load_optics_group(data['optics'])
    # dataset._load_particles(data['particles'])
    optics_groups = dataset.get_optics_group_stats()
    optics_group = optics_groups[0]
    
    box_size = optics_group['image_size']
    ang_pix = optics_group['pixel_size']
    
    max_diameter_ang = box_size * optics_group['pixel_size'] - circular_mask_thickness
    
    if part_diam is None:
        diameter_ang = box_size * 1 * optics_group['pixel_size'] - circular_mask_thickness
        print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
    else:
        if part_diam > max_diameter_ang:
            print(
                f"WARNING: Specified particle diameter {round(part_diam)} angstrom is too large\n"
                f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
            )
            diameter_ang = max_diameter_ang
        else:
            diameter_ang = part_diam
    
    if preload:
        dataset.preload_images()

    return dataset,diameter_ang,box_size,ang_pix,optics_group
   
class points2mult_volume(nn.Module): 
    def __init__(self,box_size,n_classes):
        super(points2mult_volume,self).__init__()
        self.box_size = box_size
        self.n_classes = n_classes
    
    def forward(self, points, values):
        self.batch_size = points.shape[0]
        device = points.device
        p = ((points+0.5)*(self.box_size))
        vol = torch.zeros(self.batch_size,self.n_classes,self.box_size**3).to(device)
        
        xyzpoints = p.floor().long()

        rxyz = p-xyzpoints
        
        x,y,z = xyzpoints.split(1,dim = -1)
        rx, ry, rz = rxyz.split(1,dim = -1)
        
        for dx in (0,1):
            x_ = x+dx
            wx = (1-dx)+(2*dx-1)*rx
            for dy in (0,1):
                y_ = y+dy
                wy = (1-dy)+(2*dy-1)*ry
                for dz in (0,1):
                    z_ = z+dz
                    wz = (1-dz)+(2*dz-1)*rz
                    
                    w = wx*wy*wz
                    if self.batch_size >1:
                        valid = ((0<=x_)*(x_ < self.box_size)*(0<=y_)*(y_< self.box_size)*(0<=z_)*(z_<self.box_size)).long()
                        idx = (((z_*self.box_size + y_)*self.box_size +x_)*valid).squeeze()
                        idx = torch.stack(self.n_classes*[idx],1)
                        w = (w*valid.type_as(w)).squeeze()
                        w = torch.stack(self.n_classes*[w],1)
                    else:
                        valid = ((0<=x_)*(x_ < self.box_size)*(0<=y_)*(y_< self.box_size)*(0<=z_)*(z_<self.box_size)).long()
                        idx = (((z_*self.box_size + y_)*self.box_size +x_)*valid).squeeze(2)
                        idx = torch.stack(self.n_classes*[idx],1)
                        w = (w*valid.type_as(w)).squeeze(2)
                        w = torch.stack(self.n_classes*[w],1)
                    
                    vol.scatter_add_(2,idx,w*values)
        
        vol = vol.reshape(self.batch_size,self.n_classes,self.box_size,self.box_size,self.box_size)

        return vol




def FRC(x,y,ctf,batch_reduce = 'sum'):
    y = torch.fft.fft2(torch.fft.fftshift(y.squeeze(),dim = [-1,-2]),dim=[-1,-2])
    x = torch.multiply(x,ctf)
    N = x.shape[-1]
    device = x.device
    batch_size = x.shape[0]
    eps = 1e-8
    ind = torch.linspace(-(N-1)/2,(N-1)/2-1,N)
    #end_ind = torch.round(torch.tensor(N/2)).long()
    X,Y = torch.meshgrid(ind,ind)
    R = torch.cat(batch_size*[torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2,0.5)).long()).unsqueeze(0)],0).to(device)
    num = scatter(torch.real(x*torch.conj(y)).flatten(start_dim=-2),R.flatten(start_dim=-2),reduce='mean')
    den = torch.pow(scatter(torch.abs(x.flatten(start_dim=-2))**2,R.flatten(start_dim=-2),reduce = 'mean')*scatter(torch.abs(y.flatten(start_dim=-2))**2,R.flatten(start_dim=-2),reduce = 'mean'),0.5)
    FRC = num/(den+eps)
    FRC = torch.sum(num/den,0)
    
    return FRC


def maskpoints(points,ampvar,mask,box_size):
    bs = points.shape[0]
    indpoints = torch.round((points+0.5)*(box_size-1)).long()
    indpoints = torch.clip(indpoints,max= mask.shape[-1]-1,min = 0)
    if len(indpoints.shape)>2:
        point_inds = mask[indpoints[:,:,0],indpoints[:,:,1],indpoints[:,:,2]]>0
    else:
        point_inds = mask[indpoints[:,0],indpoints[:,1],indpoints[:,2]]>0
    return points[point_inds,:], point_inds


def tensor_imshow(tensor,cmap = 'viridis'):
    x = tensor
    if len(x.shape) == 3:
        x = x[x.shape[0] // 2]

    if type(x).__module__ == 'torch':
        x = x.detach().data.cpu().numpy()
    
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(x,cmap = cmap)
    plt.axis("off")
    plt.subplots_adjust(hspace=0, wspace=0)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    matplotlib.use(backend)
    
    return fig


def tensor_plot(tensor):
    x = tensor
    if type(x).__module__ == 'torch':
        x = x.detach().data.cpu().numpy()
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x)
    
    matplotlib.use(backend)
    
    return fig


def tensor_scatter(x,y,c,s=0.1,alpha=0.5,cmap = 'jet'):

    x = x.detach().cpu()
    y = y.detach().cpu()
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.scatter(x,y,alpha = alpha,s=s,c=c,cmap = cmap)
    plt.axis("off")
    plt.subplots_adjust(hspace=0, wspace=0)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    matplotlib.use(backend)
    
    return fig

def tensor_hist(x,b):
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    a = torch.max(torch.abs(x))
    h = torch.histc(x,b,min=-a, max = a)
    h = h.detach().cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(np.arange(len(h)),h)

    matplotlib.use(backend)
    
    return fig


def visualize_latent(z,c,s=0.1,alpha=0.5,cmap = 'jet',method = 'umap'):
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    if type(z).__module__ == 'torch':
        z = z.detach().data.cpu().numpy()
    if z.shape[1]==2:
        embed = z
    elif method == 'pca':
        embed = PCA(n_components = 2).fit_transform(z)
    elif method == 'tsne':
        tsne = TSNE(n_jobs = 16)
        embed = tsne.fit_transform(z)
    elif method =='umap':
        embed = umap.UMAP(local_connectivity = 1,repulsion_strength = 2,random_state = 12).fit_transform(z)
    elif method =='projection_last':
        embed = z[:-1]
    elif method =='projection_first':
        embed = z[1:]
        
    fig, ax = plt.subplots(figsize=(5, 5))
    s = 40000/z.shape[0]
    ax.scatter(embed[:,0],embed[:,1],alpha = alpha,s=s,c=c,cmap = cmap)
    plt.axis("off")
    plt.subplots_adjust(hspace=0, wspace=0)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.set_aspect('equal', adjustable='box')
    
    matplotlib.use(backend)


    return fig

    

def points2xyz(points,title,box_size,ang_pix,types = None):

    points = box_size*ang_pix*(0.5+points.detach().data.cpu().numpy())
    atomtypes=("C",)
    atoms = np.array(['C','O','N','H','S']+['C']*200)
    ind = types.int().detach().cpu().numpy().astype(int)
    atomtypes = atoms[ind]
    f = open(title+'.xyz','a')
    f.write("%d\n%s\n" % (points.size / 3, title))
    for x, atomtype in zip(points.reshape(-1, 3), atomtypes):
        f.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))
    f.close()

def graph2bild(points,edge_index,title,color = 5):

    points = points.detach().cpu().numpy()
    f = open(title+'.bild','a')
    for k in range(points.shape[0]):
        f.write("%s %.18g %.18g %.18g %.18g\n" % ('.sphere', points[k,0], points[k,1], points[k,2],0.01))
    y = np.concatenate([points[edge_index[0].cpu().numpy()],points[edge_index[1].cpu().numpy()]],1)
    f.write('%s %.18g\n' %('.color', color))
    for k  in range(y.shape[0]):
        f.write("%s %.18g %.18g %.18g %.18g %.18g %.18g %.18g\n" % ('.cylinder',y[k,0], y[k,1], y[k,2], y[k,3], y[k,4], y[k,5], 0.6))
    f.close()


def graphs2bild(total_points,points,edge_indices,amps,title,box_size,ang_pix):
    f = open(title+'.bild','a')
    color = 8
    total_points = total_points.detach().cpu().numpy()
    total_points = (total_points+0.5)*box_size*ang_pix
    tk = 0
    for points,amps, edge_index in zip(points,amps,edge_indices):
        points = points.detach().cpu().numpy()
        points = (points+0.5)*box_size*ang_pix
        points = points/(box_size*ang_pix)-0.5
        f.write('%s %.18g\n' %('.color', color))
        if edge_index != None:
            print(points.shape)
            y = np.concatenate([total_points[edge_index[0].cpu().numpy()],total_points[edge_index[1].cpu().numpy()]],1)
            print(y.shape)
            for k  in range(y.shape[0]):
                f.write("%s %.18g %.18g %.18g %.18g %.18g %.18g %.18g\n" % ('.cylinder',y[k,0], y[k,1], y[k,2], y[k,3], y[k,4], y[k,5], 0.12))
        for k in range(points.shape[0]):
            f.write("%s %.18g %.18g %.18g %.18g\n" % ('.sphere', points[k,0], points[k,1], points[k,2],0.04*amps[tk+k]))
        color = color+10
        tk += points.shape[0]
    f.close()

def field2bild(points,field,title,box_size,ang_pix):
    f = open(title+'.bild','a')
    color = 8
    cols = torch.linalg.norm(points-field,dim = 1)
    points = points.detach().cpu().numpy()
    points = (points+0.5)*box_size*ang_pix
    field = field.detach().cpu().numpy()
    field = (field+0.5)*box_size*ang_pix
    y = np.concatenate([points,field],1)
    cols = torch.round(cols/torch.max(cols)*65).long()
    for k  in range(y.shape[0]):
        f.write('%s %.18g\n' %('.color', cols[k]))
        f.write("%s %.18g %.18g %.18g %.18g %.18g %.18g %.18g\n" % ('.arrow',y[k,0], y[k,1], y[k,2], y[k,3], y[k,4], y[k,5], 0.04))
    f.close()



def points2bild(points,amps,title,box_size,ang_pix):
    f = open(title+'.bild','a')
    color = 8
    for points,amps in zip(points,amps):
        points = points.detach().cpu().numpy()
        points = (points+0.5)*box_size*ang_pix
        f.write('%s %.18g\n' %('.color', color))
        for k in range(points.shape[0]):
            f.write("%s %.18g %.18g %.18g %.18g\n" % ('.sphere', points[k,0], points[k,1], points[k,2],0.02*amps[k]))
        color = color+20
    f.close()

def series2xyz(points,title,box_size,ang_pix):
    if type(points).__module__ == 'torch':
        points = box_size*ang_pix*(0.5+points.detach().data.cpu().numpy())
    atomtype=("C",)
    for i in range(points.shape[0]):
        print(i)
        pp = points[i].squeeze()
        f = open(title+'.xyz','a')
        f.write("%d\n%s\n" % (points[i].size / 3, title))
        for x in points.reshape(-1, 3):
            f.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))
    f.close()



def PowerSpec2(F1,batch_reduce = None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1,dim=[-2,-1])
    N = F1.shape[-1]
    ind = torch.linspace(-(N-1)/2,(N-1)/2-1,N)
    end_ind = torch.round(torch.tensor(N/2)).long()
    X,Y = torch.meshgrid(ind,ind)
    R = torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2,0.5)).long())
    res = torch.arange(start=0,end =end_ind)**2,
    p_s = scatter(torch.abs(F1.flatten(start_dim=-2)**2),R.flatten().to(F1.device),reduce = 'mean')
    if batch_reduce == 'mean':
        p_s = torch.mean(p_s,0)
    p = p_s[R]
    return p, p_s

def RadAvg2(F1,batch_reduce = None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1,dim=[-2,-1])
    N = F1.shape[-1]
    ind = torch.linspace(-(N-1)/2,(N-1)/2-1,N)
    end_ind = torch.round(torch.tensor(N/2)).long()
    X,Y = torch.meshgrid(ind,ind)
    R = torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2,0.5)).long())
    res = torch.arange(start=0,end =end_ind)**2,
    p_s = scatter(torch.abs(F1.flatten(start_dim=-2)),R.flatten().to(F1.device),reduce = 'mean')
    if batch_reduce == 'mean':
        p_s = torch.mean(p_s,0)
    p = p_s[R]
    return p, p_s

def prof2radim(w,out_value = 0):
    N = w.shape[0]
    ind = torch.linspace(-N,N-1,2*N)
    X,Y = torch.meshgrid(ind,ind)
    R = torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2,0.5)).long())
    R[R>N-1]=N-1
    W = w[R]
    W[R == N-1] = out_value
    return W
    

def RadialAvg(F1,batch_reduce = None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1,dim=[-3,-2,-1])
    N = F1.shape[-1]
    ind = torch.linspace(-(N-1)/2,(N-1)/2-1,N)
    end_ind = torch.round(torch.tensor(N/2)).long()
    X,Y,Z = torch.meshgrid(ind,ind,ind)
    R = torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2+Z**2,0.5)).long())
    res = torch.arange(start=0,end =end_ind)**2,
    
    if len(F1.shape)==3:
        p_s = scatter(torch.abs(F1.flatten(start_dim=-3)),R.flatten(),reduce = 'mean')
    return p_s

def RadialAvgProfile(F1,batch_reduce = None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1,dim=[-3,-2,-1])
    device = F1.device
    print(device)
    N = F1.shape[-1]
    ind = torch.linspace(-(N-1)/2,(N-1)/2-1,N)
    end_ind = torch.round(torch.tensor(N/2)).long()
    X,Y,Z = torch.meshgrid(ind,ind,ind)
    R = torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2+Z**2,0.5)).long()).to(device)
    res = torch.arange(start=0,end =end_ind)**2,

    if len(F1.shape)==3:
        p_s = scatter(torch.abs(F1.flatten(start_dim=-3)),R.flatten(),reduce = 'mean')
    Prof = torch.zeros_like(R).float().to(device)
    Prof[R] = p_s[R]
    
    return Prof


def fourier_shift_2d(
        grid_ft,
        xshift,
        yshift
):
    s = grid_ft.shape[-1]
    xshift = -xshift / float(s)
    yshift = -yshift / float(s)

    if torch.is_tensor(grid_ft):
        ls = torch.linspace(-s // 2, s // 2 - 1, s)
        y, x = torch.meshgrid(ls, ls)
        x = x.to(grid_ft.device)
        y = y.to(grid_ft.device)
        dot_prod = 2 * np.pi * (x[None, :, :] * xshift[:, None, None] + y[None, :, :] * yshift[:, None, None])
        dot_prod = torch.fft.fftshift(dot_prod,dim=[-1,-2])
        a = torch.cos(dot_prod)
        b = torch.sin(dot_prod)
    else:
        ls = np.linspace(-s // 2, s // 2 - 1, s),
        y, x = np.meshgrid(ls, ls, indexing="ij")
        dot_prod = 2 * np.pi * (x[None, :, :] * xshift[:, None, None] + y[None, :, :] * yshift[:, None, None])
        dot_prod = torch.fft.fftshift(dot_prod,dim=[-1,-2])
        a = np.cos(dot_prod)
        b = np.sin(dot_prod)

    ar = a * grid_ft.real
    bi = b * grid_ft.imag
    ab_ri = (a + b) * (grid_ft.real + grid_ft.imag)

    return ar - bi + 1j * (ab_ri - ar - bi)


def FlipZ(F1):
    Fz = torch.flip(F1,[-3])
    return(Fz)

def PowerSpec(F1,batch_reduce = None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1,dim=[-3,-2,-1])
    N = F1.shape[-1]
    ind = torch.linspace(-(N-1)/2,(N-1)/2-1,N)
    end_ind = torch.round(torch.tensor(N/2)).long()
    X,Y,Z = torch.meshgrid(ind,ind,ind)
    R = torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2+Z**2,0.5)).long())
    res = torch.arange(start=0,end =end_ind)**2,
    
    if len(F1.shape)==3:
        p_s = scatter(torch.abs(F1.flatten(start_dim=-3))**2,R.flatten(),reduce = 'sum')
    return p_s[:end_ind],res[0]


def FSC(F1,F2,ang_pix = 1, visualize = False):
    device = F1.device
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1,dim=[-3,-2,-1])
    if F2.is_complex():
        pass
    else:
        F2 = torch.fft.fftn(F2,dim=[-3,-2,-1])
    
    if F1.shape != F2.shape:
        print('The volumes have to be the same size')
    
    N = F1.shape[-1]
    ind = torch.linspace(-(N-1)/2,(N-1)/2-1,N)
    end_ind = torch.round(torch.tensor(N/2)).long()
    X,Y,Z = torch.meshgrid(ind,ind,ind)
    R = torch.fft.fftshift(torch.round(torch.pow(X**2+Y**2+Z**2,0.5)).long()).to(device)

    
    if len(F1.shape)==3:
        num = torch.zeros(torch.max(R)+1).to(device)
        den1 = torch.zeros(torch.max(R)+1).to(device)
        den2 = torch.zeros(torch.max(R)+1).to(device)
        num.scatter_add_(0,R.flatten(),torch.real(F1*torch.conj(F2)).flatten())
        den = torch.pow(den1.scatter_add_(0,R.flatten(),torch.abs(F1.flatten(start_dim=-3))**2)*den2.scatter_add_(0,R.flatten(),torch.abs(F2.flatten(start_dim=-3))**2),0.5)
        FSC = num/den
    res = N*ang_pix/torch.arange(end_ind)
    FSC[0]=1
    if visualize == True:
        plt.figure(figsize=(10,10))
        plt.rcParams['axes.xmargin'] = 0
        plt.plot(FSC[:end_ind].cpu(),c='r') 
        plt.plot(torch.ones(end_ind)*0.5,c='black',linestyle = 'dashed')
        plt.plot(torch.ones(end_ind)*0.143,c='slategrey',linestyle = 'dotted')
        plt.xticks(torch.arange(start=0,end = end_ind,step = 10),labels = np.round(res[torch.arange(start=0,end = end_ind,step = 10)].numpy(),1))
        plt.show()
    return FSC[0:end_ind],res



def make_color_map(p,mean_dist,kernel,box_size,device,n_classes):
    p2V = points2mult_volume(box_size, n_classes)
    p = p.unsqueeze(0)
    mean_dist = mean_dist.unsqueeze(0).to(device)
    Cmap = p2V(p.expand(2,-1,-1),mean_dist.expand(2,-1)).to(device)
    return Cmap



def select_subset(starfile,subset_indices,outname):
    with open(starfile,'r+') as f:
        with open(outname,'w') as output:
            d = f.readlines()
            mode = 'handlers'
            count = 0
            startcount = 0
            for i in d:
                if i.startswith('data_particles'):
                    print('particles')
                    mode = 'particles'
                if mode == 'handlers':
                    output.write(i)

                if mode == 'particles':
                    output.write(i)
                    if i.startswith('loop_'):
                        print('particles_loop')
                        mode = 'particle_props'

                if mode == 'particle_props':
                    if i.startswith('_'):
                        output.write(i)
                    else:
                        if startcount == 0:
                            startcount = count
                        if count-startcount in subset_indices:
                            output.write(i)
                count = count+1
            output.close()

def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'weight' not in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def pdb2points(name,random = False):
    
    pdb = PDBParser()
    cif = MMCIFParser()
    
    total_length = 0
    total_gr = []

    
    if name.endswith('.pdb'):
        model = pdb.get_structure('model',name)
    elif name.endswith('.cif'):
        model = cif.get_structure('model',name)

    coords = []
    for chain in model.get_chains():
        residues = chain.get_residues()
        for res in residues:
            for a in res.get_atoms():
                coords.append(a.get_coord())

    coords = torch.tensor(np.array(coords))

    return coords


def points2pdb(name,outname,points,random = False):
    if name.endswith('.pdb'):
        io = PDBIO()
    elif name.endswith('.cif'):
        io = MMCIFIO()
    pdb = PDBParser()
    cif = MMCIFParser()
    
    total_length = 0
    total_gr = []

    
    if name.endswith('.pdb'):
        model = pdb.get_structure('model',name)
    elif name.endswith('.cif'):
        model = cif.get_structure('model',name)
    i = 0
    coords = []
    for chain in model.get_chains():
        residues = chain.get_residues()
        for res in residues:
            for a in res.get_atoms():
                a.set_coord(points[i])
                i +=1

    io.set_structure(model)
    io.save(outname,preserve_atom_numbering = True)




def pdb2graph(name):
    
    pdb = PDBParser()
    cif = MMCIFParser()
    
    total_length = 0
    total_gr = []
    total_amp = []


    if name.endswith('.pdb'):
        model = pdb.get_structure('model',name)
    elif name.endswith('.cif'):
        model = cif.get_structure('model',name)

    
    for chain in model.get_chains():
        coords = []
        amp = []
        direction = []
        residues = chain.get_residues()
        for res in residues:
            cm = res.center_of_mass()
            try:
                coords.append(res['CA'].get_coord())
                direction.append(cm-res['CA'].get_coord())
                amp.append(len(res.child_list))
                amp.append(len(res.child_list))
            except:
                coords.append(res.center_of_mass())
                direction.append(np.zeros(3))
                amp.append(len(res.child_list))
                amp.append(len(res.child_list))
                

        coords = torch.tensor(np.array(coords))
        amp = torch.tensor(np.array(amp))
        direction = torch.tensor(np.array(direction))

        if total_length ==0:
            gr = torch.stack([torch.arange(start = 0,end = coords.shape[0]-1),torch.arange(start = 1,end = coords.shape[0])],0)
            gr = torch.cat([gr,torch.tensor([coords.shape[0]-1,coords.shape[0]-1]).unsqueeze(1)],1)
        else:
            gr_c = torch.stack([torch.arange(start = total_length,end = total_length+coords.shape[0]-1),torch.arange(start = total_length+1,end = total_length+coords.shape[0])],0)
            gr_c = torch.cat([gr_c,torch.tensor([total_length+coords.shape[0]-1,total_length+coords.shape[0]-1]).unsqueeze(1)],1)
            gr = torch.cat([gr,gr_c],1)
        if total_length == 0:
            total_coords = coords
            total_dirs = direction
            total_amp = amp
        else:

            total_coords = torch.cat([total_coords,coords],0)
            total_dirs = torch.cat([total_dirs,direction],0)
            total_amp = torch.cat([total_amp,amp],0)
        total_length += coords.shape[0]
        
    
    diff1 = total_coords[gr[0]]-total_coords[gr[1]]
    diffnorm1 = torch.linalg.norm(diff1,dim = 1)
    diff = total_coords[gr[0]]-total_coords[gr[1]]
    gr = gr[:,diffnorm1<7]
    randc = torch.randn_like(total_coords)
    randc /= torch.stack(3*[torch.linalg.norm(randc,dim = 1)],1)
    zero_inds = torch.linalg.norm(diff, dim = 1) == 0
    diff[zero_inds == True] = diff[torch.roll(zero_inds == True, -1)]
    #norms = torch.cross(direction,randc)
    norms = total_dirs.float()
    zinds = torch.where(torch.linalg.norm(norms,dim = 1)==0)
    norms[zinds[0]] = torch.cross(diff[zinds[0]],randc[zinds[0]]).float()
    norms = norms/torch.stack(3*[torch.linalg.norm(norms,dim = 1)],1)*3.7
    add_coords = total_coords+norms


    add1 = torch.arange(start = 0, end = len(total_coords))
    add2 = torch.arange(start = len(total_coords), end = len(total_coords)+len(add_coords))

    gr_add = torch.stack([add1,add2],0)
    gr = torch.cat([gr,gr_add],1)
    
    xyz = torch.cat([total_coords,add_coords],0)
    gr =gr[:, gr[0] != gr[1]]
    gr = gr.long()
        
    return xyz, gr, total_amp

def pdb2allatoms(names,box_size,ang_pix):
    t_positions = []
    atoms = []
    for name in names:
        atompos = pdb2points(name)
        atoms.append(atompos)
    atom_positions = torch.cat(atoms,0)
    atom_positions = atom_positions/(box_size*ang_pix)
    if torch.min(atom_positions)>0: #correct for normal pdbs
        atom_positions = atom_positions-0.5
    
    return atom_positions

def initial_optimization(cons_model, atom_model,device,directory,angpix, N_epochs):
    z0 = torch.zeros(2, 2)
    r0 = torch.zeros(2, 3)
    t0 = torch.zeros(2,2)
    V0 = atom_model.volume(r0.to(device),t0.to(device))
    V0 = V0[0].float()
    box_size = V0.shape[0]
    with mrcfile.new(directory+'/optimization_volume.mrc',overwrite=True) as mrc:
        mrc.set_data(V0.detach().cpu().numpy())
    atom_model.requires_grad = False
    cons_model.pos.requires_grad = False
    coarse_params = cons_model.parameters()
    coarse_optimizer = torch.optim.Adam(coarse_params, lr=0.001)
    cons_model.amp.requires_grad = True
    V = cons_model.volume(r0.to(device),t0.to(device)).float()
    fsc,res=FSC(V0.detach(),V[0],1,visualize = False)
    
    for i in tqdm(range(N_epochs)):
        coarse_optimizer.zero_grad()
        V = cons_model.volume(r0.to(device),t0.to(device)).float()
        fsc,res=FSC(V0.detach(),V[0],1,visualize = False)
        loss = -torch.sum(fsc) #1e-2*f1(lay(coarse_model.pos))
        types = torch.argmax(torch.nn.functional.softmax(cons_model.ampvar, dim=0), dim=0)
        points2xyz(cons_model.pos, '/cephfs/schwab/approximation/positions' + str(i).zfill(3), box_size, angpix, types)
        #print(torch.nn.functional.mse_loss(V[0],V0.detach()).detach())
        #print(f1(lay(coarse_model.pos)).detach())
        loss.backward()
        coarse_optimizer.step()
        
    with mrcfile.new(directory+'/coarse_initial_volume.mrc',overwrite=True) as mrc:
        mrc.set_data(V[0].detach().cpu().numpy())

    print('Total FSC value:', torch.sum(fsc))
    
def load_models(path,device,box_size,n_classes):
    cp = torch.load(path,map_location=device)
    encoder_half1 = cp['encoder_half1']
    encoder_half2 = cp['encoder_half2']
    #cons_model_l = cp['consensus']
    deformation_half1 = cp['decoder_half1']
    deformation_half1.p2i = points2mult_image(box_size,n_classes,1)
    deformation_half1.i2F = ims2F_form(box_size,device,n_classes,1)
    deformation_half2 = cp['decoder_half2']
    deformation_half2.p2i = points2mult_image(box_size,n_classes,1)
    deformation_half2.i2F = ims2F_form(box_size,device,n_classes,1)
    poses = cp['poses']
    encoder_half1.load_state_dict(cp['encoder_half1_state_dict'])
    encoder_half2.load_state_dict(cp['encoder_half2_state_dict'])
    #cons_model.load_state_dict(cp['consensus_state_dict'])
    deformation_half1.load_state_dict(cp['decoder_half1_state_dict'])
    deformation_half2.load_state_dict(cp['decoder_half2_state_dict'])
    poses.load_state_dict(cp['poses_state_dict'])
    #cons_model.p2i.device = device
    deformation_half1.p2i.device = device
    deformation_half2.p2i.device = device
    #cons_model.proj.device = device
    deformation_half1.proj.device = device
    deformation_half2.proj.device = device
    #cons_model.i2F.device = device
    deformation_half1.i2F.device = device
    deformation_half2.i2F.device = device
    #cons_model.p2v.device = device
    deformation_half1.p2v.device = device
    deformation_half1.device = device
    deformation_half2.p2v.device = device
    deformation_half2.device = device
    #cons_model.device = device
    deformation_half1.to(device)
    deformation_half2.to(device)
    #cons_model.to(device)
    
    return encoder_half1, encoder_half2, deformation_half1, deformation_half2
    
    
def reset_all_linear_layer_weights(model: nn.Module) -> nn.Module:
    """
    Resets all weights recursively for linear layers.

    ref:
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Linear:
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(init_weights)
    
    
class spatial_grad(nn.Module):
    #For TV regularization
    def __init__(self,spatial_grad,box_size):
        super(spatial_grad,self).__init__()
        self.box_size = box_size
    
    def forward(self,x):
        batch_size = x.size()[0]
        t_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]

        t_grad = (x[:,:,2:,:,:]-x[:,:,:h_x-2,:,:])
        h_grad = (x[:,:,:,2:,:]-x[:,:,:,:h_x-2,:])
        w_grad = (x[:,:,:,:,2:]-x[:,:,:,:,:w_x-2])
        t_grad = torch.nn.functional.pad(t_grad[0,0],[0,0,0,0,1,1])
        h_grad = torch.nn.functional.pad(h_grad[0,0],[0,0,1,1,0,0])
        w_grad = torch.nn.functional.pad(w_grad[0,0],[1,1,0,0,0,0])
        grad = torch.stack([t_grad,h_grad,w_grad],0)
        return grad

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def compute_threshold(V):
    Vmin = torch.min(V)
    Vmax = torch.max(V)
    V_norm = torch.sum(V**2)
    lam = torch.linspace(Vmin,Vmax,100)
    percent = torch.zeros_like(lam)
    for i in range(100):
        Vi = torch.sum(V[V>lam[i]]**2)
        percent[i] = Vi/V_norm
    th_ind = torch.argmin(torch.abs(percent-0.91))
    return lam[th_ind]

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def make_equidistant(x, y, N):
    N_p = x.shape[0]
    points = np.stack([x, y], 1)
    dp = points[1:] - points[:-1, :]
    n_points = []
    dists = np.linalg.norm(dp, axis=1)
    min_dist = np.min(dists)
    curve_length = np.sum(dists)
    seg_length = curve_length / (N_p - 1)
    print('Curve length:', curve_length)
    for i in range(N_p - 1):
        q = dists[i] / min_dist
        Nq = np.round(q)
        for j in range(int(Nq) - 1):
            n_points.append(points[i] + j * min_dist * (dp[i] / np.linalg.norm(dp[i])))
    n_points = np.array(n_points)
    plt.scatter(n_points[:, 0], n_points[:, 1])
    plt.show()
    frac = np.maximum(int(np.round(n_points.shape[0] / N)), 1)
    return n_points[::frac, :], points
