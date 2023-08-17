#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:30:53 2023

@author: schwab
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import init, ReflectionPad2d, ZeroPad2d
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from unet_blocks import *
import numpy as np
import matplotlib.pyplot as plt
import gc

torch.manual_seed(0)
# U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/abs/1505.04597
data = np.load("/cephfs/schwab/ML_test/stroke/data/train_data.npz")
#data = np.load("C:/Users/nadja/Documents/Stroke/data/train_data.npz")

X = data["X"][:]
X = torch.tensor(X).unsqueeze(1)
Y = data["Y"][:]
Y = torch.tensor(Y).unsqueeze(1)
Y = torch.tensor(Y)
L = data["L"][:]
L = torch.tensor(L[:, :, :, :1]).unsqueeze(1)
L = torch.tensor(L)
del(data)

data = torch.cat([X, Y, L], axis=1)
print(data.shape)

del(X)
del(Y)
del(L)
gc.collect()
data_val = np.load("/cephfs/schwab/ML_test/stroke/data/val_data.npz")
#data_val = np.load("C:/Users/nadja/Documents/Stroke/data/val_data.npz")

X = data_val["X"][:]
X = torch.tensor(X).unsqueeze(1)
Y = data_val["Y"][:]
Y = torch.tensor(Y).unsqueeze(1)
Y = torch.tensor(Y)
L = data_val["L"][:]
L = torch.tensor(L[:, :, :, :1]).unsqueeze(1)
L = torch.tensor(L)
gc.collect()
del(data_val)

data_val = torch.cat([X, Y, L], axis=1)
del(X)
del(Y)
del(L)
gc.collect()


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AutoEncoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Joint_reconstruction_and_segmentation_dwi_lesion:
    def __init__(
        self,
        learning_rate: float = 5e-6,
        device: str = 'cuda:0',
        inputs: str = "DWI"
    ):
        self.mean_FP = []
        self.mean_FN = []
        self.Recall = 0
        self.mean_recall = []
        self.learning_rate = 5e-6
        self.net = UNet(1, 1)
        self.recon_weight = 0.5
        self.alpha = 0.99
        self.gamma = 0.5
        self.delta = 0.5
        self.weight = 0.001
        self.lesion_yes_no = []
        #self.inputs = "DWI"
        self.dwi_2_adc_net = AutoEncoder(n_channels=1, n_classes=1).to(device)
        self.optimizer_dwi_2_adc = optim.Adam(
            self.dwi_2_adc_net.parameters(), lr=self.learning_rate)
        self.segmentation_net = UNet(n_channels=1, n_classes=1).to(device)
        self.optimizer = optim.Adam(
            self.segmentation_net.parameters(), lr=self.learning_rate)
        self.data = data
        self.data_val = data_val
        self.gt_val = data_val[:, 2]
        self.gt = data[:, 2]
        self.device = 'cuda:0'
        self.Dice = 0
        self.Dice_isles = 0
        self.inputs = inputs
        self.mean_dice = []
        self.mean_dice_isles = []
        self.FP = 0
        self.FN = 0
        self.mean_spec = []
        self.Spec = 0
        self.max_mean_dice = 0

    def init_NW(self, device):
        if self.inputs == "DWI":
            self.segmentation_net = UNet(n_channels=1, n_classes=1).to(device)
        if self.inputs == "DWI + ADC":
            self.segmentation_net = UNet(n_channels=2, n_classes=1).to(device)
        if self.inputs == "DWI + ADC + DWI*ADC":
            self.segmentation_net = UNet(n_channels=3, n_classes=1).to(device)
        self.optimizer = optim.Adam(
            self.segmentation_net.parameters(), lr=self.learning_rate)

    def normalize(self, f):
        '''normalize image to range [0,1]'''
        f = torch.tensor(f).float()
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
        return f

    def define_nonlesion_slices(self, segmentation_mask):
        '''function that give back a tensor indicating whether on slice i there is a lesion or not'''
        lesion_yes_no = torch.empty((len(segmentation_mask), 1))
        # if there is no lesion on the slice,
        index = torch.tensor([[x] for x in range(
            len(segmentation_mask)) if torch.max(segmentation_mask[x]) == 0])

        if len(index) > 0:
            lesion_yes_no[index] = 1
        else:
            lesion_yes_no == torch.zeros_like(lesion_yes_no)
        lesion_yes_no = torch.round(lesion_yes_no)

        return lesion_yes_no

    def compute_weight(self):
        '''use the labels of the whole dataset and compute imbalance for BCE loss term'''
        shape = self.gt.shape
        self.weight = torch.sum(self.gt)/(shape[2]**2*shape[0])

    def segmentation_step(self, data, data_val):
        Data_loader = DataLoader(
            self.data,  batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
        Data_loader_val = DataLoader(self.data_val,
                                     batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
        del(self.data)
        del(data)
      #  del(self.data_val)
     #   del(data_val)
        gc.collect()
        losses = []
        i = 0
        running_loss = .0
        for epoch in range(500):
            for features in Data_loader:
                # explanation: features[:,:1] is DWI, features[:,1:2] is ADC, features [:,2:3] gt
                if self.inputs == "DWI":
                    input_x = features[:, 0:1, :, :, 0]
                    dwi_input = features[:, 0:1, :, :, 0]
                    adc_gt = features[:, 1:2, :, :, 0].to(self.device)
                    segmentation_mask = features[:,
                                                 2:, :, :, 0].to(self.device)

                if self.inputs == "DWI + ADC":
                    input_x = features[:, :2, :, :, 0]
                    dwi_input = features[:, 0:1, :, :, 0]
                    adc_gt = features[:, 1:2, :, :, 0].to(self.device)
                    segmentation_mask = features[:,
                                                 2:, :, :, 0].to(self.device)

                if self.inputs == "DWI + ADC + DWI*ADC":
                    dwi_input = features[:, 0:1, :, :, 0]
                    adc_gt = features[:, 1:2, :, :, 0].to(self.device)
                    segmentation_mask = features[:,
                                                 2:, :, :, 0].to(self.device)

                    input_x = torch.cat([features[:, 0:1, :, :, 0], features[:, 1:2, :, :, 0],
                                        features[:, :1, :, :, 0]*features[:, 1:2, :, :, 0]], axis=1)
                # print(input_x.shape)
                self.optimizer.zero_grad()
                # initialize optim for dwi to adc nw
                self.optimizer_dwi_2_adc.zero_grad()
                # gt for segmentation
                # gt for ADC to dwi nw training
                is_lesion_on_slice = self.define_nonlesion_slices(
                    segmentation_mask).to(self.device)
                # segmentation predictions
                output = F.sigmoid(self.segmentation_net(
                    input_x.float().to(self.device)))
                loss_seg = self.joint_loss(segmentation_mask, output)
                output_dwi_2_adc = self.dwi_2_adc_net(
                    dwi_input.float().to(self.device))
                # only for learning DWI to ADC, no CV included
                loss_reco = self.DWI2ADC_loss_only_difference(
                    adc_gt, output_dwi_2_adc, output, segmentation_mask, is_lesion_on_slice)
                # losses.append(loss.item())
                #### if there are lesions on at least one slice################
                # if torch.max(is_lesion_on_slice) <0.5 or epoch<=50:

                if epoch <= 50:
                    loss_total = loss_seg + loss_reco
                    # defince the total loss used for training
                    loss_total.backward()
                    self.optimizer.step()
                    self.optimizer_dwi_2_adc.step()
                    print("segmentation_loss:"+str(loss_seg))

                else:
                    loss_dwi_2_adc = self.DWI2ADC_loss(
                        adc_gt, output_dwi_2_adc, output, segmentation_mask, is_lesion_on_slice)
                    loss_total = loss_seg + 0.1*loss_dwi_2_adc
                    loss_total.backward()
                    self.optimizer.step()
                    self.optimizer_dwi_2_adc.step()
                    print("segmentation_loss:"+str(loss_seg))
                    print("DWI to ADC loss:"+str(loss_dwi_2_adc))
                # dwi 2 adc prediction

                with torch.no_grad():
                    running_loss += loss_total.item()
                    if i % 10 == 9:
                        print('[Epoque : %d, iteration: %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0
                    i += 1
            # plt.subplot(1,2,1)
            # plt.imshow(torch.round(output[0][0].detach().cpu()))
            # plt.subplot(1,2,2)
            # plt.imshow(input_x[0][0].detach().cpu())
            # plt.show()

            with torch.no_grad():
                for features in Data_loader_val:
                    if self.inputs == "DWI":
                        input_x = features[:, 0:1, :, :, 0]
                        dwi_input = features[:, 0:1, :, :, 0]
                        adc_gt = features[:, 1:2, :, :, 0]
                        segmentation_mask = features[:, 2:, :, :, 0].to(
                            self.device)

                    if self.inputs == "DWI + ADC":
                        input_x = features[:, :2, :, :, 0]
                        dwi_input = features[:, 0:1, :, :, 0]
                        adc_gt = features[:, 1:2, :, :, 0]
                        segmentation_mask = features[:, 2:, :, :, 0].to(
                            self.device)

                    if self.inputs == "DWI + ADC + DWI*ADC":
                        dwi_input = features[:, 0:1, :, :, 0]
                        adc_gt = features[:, 1:2, :, :, 0]
                        segmentation_mask = features[:, 2:, :, :, 0].to(
                            self.device)

                        input_x = torch.cat([features[:, 0:1, :, :, 0], features[:, 1:2, :, :, 0],
                                            features[:, :1, :, :, 0]*features[:, 1:2, :, :, 0]], axis=1)
                    # print(input_x.shape)
                    output = F.sigmoid(self.segmentation_net(
                        input_x.float().to(self.device)))
                    output_dwi_2_adc = self.dwi_2_adc_net(
                        dwi_input.float().to(self.device))

                    self.evaluate(segmentation_mask.to(self.device), output)
                plt.subplot(3, 2, 1)
                plt.imshow(torch.round(output[0][0].detach().cpu()))
                plt.subplot(3, 2, 2)
                plt.imshow(input_x[0][0].detach().cpu())
                plt.subplot(3, 2, 3)
                plt.imshow(output_dwi_2_adc[0][0].detach().cpu())
                plt.subplot(3, 2, 4)
                plt.imshow(adc_gt[0][0]-output_dwi_2_adc[0]
                           [0].detach().cpu(), cmap="inferno")
                plt.colorbar()
                plt.subplot(3, 2, 5)
                plt.imshow(adc_gt[0][0].detach().cpu(), cmap="inferno")
                plt.colorbar()
                plt.subplot(326)
                plt.imshow(segmentation_mask[0][0].detach().cpu())
                plt.show()

            self.mean_dice.append(
                ((torch.tensor(self.Dice))/len(Data_loader_val)).detach().cpu())
            self.mean_dice_isles.append(
                ((torch.tensor(self.Dice_isles))/len(Data_loader_val)).detach().cpu())
            self.mean_recall.append(
                ((torch.tensor(self.Recall))/len(Data_loader_val)).detach().cpu())
            self.mean_spec.append(
                ((torch.tensor(self.Spec))/len(Data_loader_val)).detach().cpu())
            self.mean_FP.append(
                ((torch.tensor(self.FP))//len(Data_loader_val)).detach().cpu())
            self.mean_FN.append(
                ((torch.tensor(self.FN))/len(Data_loader_val)).detach().cpu())

            if self.mean_dice_isles[-1] > self.max_mean_dice:
                self.max_mean_dice = self.mean_dice_isles[-1]
                name_weights = "weights_joint_"+self.inputs+".hdf5"
                # torch.save(mynet.segmentation_net.state_dict(),
                #            "C:/Users/nadja/Documents/Stroke/weights/"+name_weights)
                torch.save(mynet.segmentation_net.state_dict(),
                           "/cephfs/schwab/ML_test/stroke/weights/"+name_weights)
                print('saved weights')

            self.Dice = 0
            self.Dice_isles = 0
            self.Recall = 0
            self.Spec = 0
            self.FP = 0
            self.FN = 0

            plt.plot(self.mean_dice, label="val_dice")
            plt.plot(self.mean_dice_isles, label="val_dice_isles")
            plt.legend()
            plt.show()

            plt.plot(self.mean_recall, label="Recall")
            plt.legend()
            plt.show()

            plt.plot(self.mean_spec, label="Specifity")
            plt.legend()
            plt.show()

            plt.plot(self.mean_FN, label="val_Fn")
            plt.plot(self.mean_FP, label="val_FP")
            plt.legend()
            plt.show()
            del output
            gc.collect()
            del segmentation_mask
            gc.collect()

            # self.free_gpu_cache()


########### tversky loss ####################################


    def tversky(self, tp, fn, fp):
        loss2 = 1 - ((torch.sum(tp)+0.000001)/((torch.sum(tp) +
                     self.gamma*torch.sum(fn) + self.delta*torch.sum(fp)+0.000001)))
        return loss2

    def evaluate(self, segmentation_mask, output):
        output = torch.round(output)
        tp = torch.sum(output*segmentation_mask)
        tn = torch.sum((1-output)*(1-segmentation_mask))
        fn = torch.sum((1-output)*(segmentation_mask))
        fp = torch.sum((output)*(1-segmentation_mask))
        Recall = (tp+0.0001)/(tp + fn + 0.0001)
        Spec = (tn / (tn + fp))
        Dice = 2*tp/(2*tp + fn + fp)

        im1 = np.asarray(segmentation_mask.detach().cpu()).astype(bool)
        im2 = np.asarray(output.detach().cpu()).astype(bool)

        if im1.shape != im2.shape:
            raise ValueError(
                "Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return 1.0

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        dice_isles = 2.0 * intersection.sum() / im_sum

        self.Dice_isles += dice_isles
        self.Dice += Dice.detach().cpu()
        self.FP += fp.detach().cpu()
        self.FN += fn.detach().cpu()
        self.Recall += Recall
        self.Spec += Spec
        del(fn, fp, tn, tp)
        gc.collect()


###### loss for joint reconstruction and segmentation ########

    def joint_loss(self, segmentation_mask, output):
        weights = torch.stack(
            [torch.tensor(1-self.weight), torch.tensor(self.weight)]).to(self.device)
        output = torch.stack([output, 1-output], axis=-1)
        segmentation_mask = torch.stack(
            [segmentation_mask, 1-segmentation_mask], axis=-1)

       # weights = 1-torch.tensor(self.weight)
        output = torch.clip(output, min=1e-6)
        loss1 = -torch.sum(segmentation_mask *
                           torch.log(output) * weights, axis=-1)
        loss1 = torch.mean(loss1)
        '''tversky preperation'''
        y_true_f = torch.flatten(segmentation_mask[:, :, :, :, :1])
        y_pred_f = torch.flatten(output[:, :, :, :, :1])
        fp = (1-y_true_f)*y_pred_f
        fn = (1-y_pred_f)*y_true_f
        tp = y_pred_f*y_true_f
        loss = (1-self.recon_weight)*(self.alpha*loss1) + \
            (1-self.alpha)*(self.tversky(tp, fn, fp))
        # loss = self.tversky(tp,fn,fp
        del(tp, fp, fn, y_true_f, y_pred_f)
        gc.collect()
        return loss


############# loss function for DWI to ADC learning ####################
    ''''as input, we have the real ADC, the reconstructed ADC,and  the segmentation mask '''

    def DWI2ADC_loss(self, adc_gt, output_dwi_2_adc, output, segmentation_mask, is_lesion_on_slice):

        #shape = adc_gt.shape
        #loss_bg = torch.sum(is_lesion_on_slice.unsqueeze(-1).unsqueeze(-1) * (adc_gt-output_dwi_2_adc)**2*(1-output))/torch.sum((shape[2]**2*is_lesion_on_slice))
        #loss_fg = torch.sum(is_lesion_on_slice.unsqueeze(-1).unsqueeze(-1) * (adc_gt-output_dwi_2_adc)**2*(1-output))/torch.sum((shape[2]**2*is_lesion_on_slice))
        loss_bg = torch.sum((adc_gt-output_dwi_2_adc)**2 *
                            (1-output))/torch.sum((1-output))
        # this is the CV constant term for brain inarct region
        loss_fg = torch.sum((adc_gt-torch.sum(adc_gt*output) /
                            torch.sum(output))**2*(output))/torch.sum(output)
        return loss_bg + loss_fg

    ''''as input, we have the real ADC, the reconstructed ADC,and  the segmentation mask '''

    def DWI2ADC_loss_only_difference(self, adc_gt, output_dwi_2_adc, output, segmentation_mask, is_lesion_on_slice):
        if torch.sum(is_lesion_on_slice) > 0:
            shape = adc_gt.shape
            loss_bg = torch.sum(is_lesion_on_slice.unsqueeze(-1).unsqueeze(-1) * (
                adc_gt-output_dwi_2_adc)**2)/torch.sum((shape[2]**2*torch.sum(is_lesion_on_slice)))
        else:
            loss_bg = 0.
       # loss_bg = torch.sum((adc_gt-output_dwi_2_adc)**2*(1-output))/torch.sum((1-output))
        # this is the CV constant term for brain inarct region
        #loss_fg = torch.sum((adc_gt-torch.sum(adc_gt*output)/torch.sum(output))**2*(output))/torch.sum(output)
        return loss_bg


mynet = Joint_reconstruction_and_segmentation_dwi_lesion(
    inputs="DWI")
mynet.compute_weight()
mynet.init_NW(device=mynet.device)
mynet.segmentation_step(data, data_val)
