#!/usr/bin/env python3

"""
Module for managing pytorch dataset of particles.
"""

import os
import warnings
from typing import Any, List, Dict

import numpy as np
import mrcfile
from scipy.ndimage import shift

import torch
from torch.utils.data import Dataset
from .ctf import ContrastTransferFunction


class ParticleDataset(Dataset):
    def __init__(self) -> None:
        self.image_file_paths = None

        self.part_rotation = None
        self.part_translation = None
        self.part_defocus = None
        self.part_og_idx = None
        self.part_stack_idx = None
        self.part_image_file_path_idx = None
        self.part_norm_correction = None
        self.part_noise_group_id = None
        self.part_preloaded_image = None

        self.has_ctf = None
        self.compute_ctf = True

        # Dictionaries mapping optics group id to data
        self.optics_group_stats = []
        self.optics_group_ctfs = []

    def initialize(
            self,
            image_file_paths: List,
            part_rotation: List,
            part_translation: List,
            part_defocus: List,
            part_og_idx: List,
            part_stack_idx: List,
            part_image_file_path_idx: List,
            part_norm_correction: List,
            part_noise_group_id: List,
            optics_group_stats: List
    ) -> None:
        self.image_file_paths = image_file_paths
        self.part_rotation = part_rotation
        self.part_translation = part_translation
        self.part_defocus = part_defocus
        self.part_og_idx = part_og_idx
        self.part_stack_idx = part_stack_idx
        self.part_image_file_path_idx = part_image_file_path_idx
        self.part_norm_correction = part_norm_correction
        self.part_noise_group_id = part_noise_group_id
        self.optics_group_stats = optics_group_stats

        if np.all(np.isnan(self.part_defocus)):
            self.has_ctf = False
        else:
            self.has_ctf = True
            self.setup_ctfs()

    def setup_ctfs(self, h_sym: bool = False, compute_ctf: bool = None):
        if self.part_defocus is None:
            return

        if compute_ctf is not None:
            self.compute_ctf = compute_ctf

        for og in self.optics_group_stats:
            if og["voltage"] is not None or \
                    og["spherical_aberration"] is not None or \
                    og["amplitude_contrast"] is not None:
                ctf = ContrastTransferFunction(
                    og["voltage"],
                    og["spherical_aberration"],
                    og["amplitude_contrast"]
                )
            else:
                ctf = None
                warnings.warn(f"WARNING: CTF parameters missing for optics group ID: {id}", RuntimeWarning)

            self.optics_group_ctfs.append(ctf)

    def get_optics_group_stats(self):
        return self.optics_group_stats

    def get_optics_group_ctfs(self):
        return self.optics_group_ctfs

    def preload_images(self):
        self.part_preloaded_image = [None for _ in range(len(self.part_rotation))]
        part_index_list = np.arange(len(self.part_rotation))
        unique_file_idx, unique_reverse = np.unique(self.part_image_file_path_idx, return_inverse=True)
        for i in range(len(unique_file_idx)):
            file_idx = unique_file_idx[i]
            path = self.image_file_paths[file_idx]
            mrc = mrcfile.open(path, 'r')

            this_file_mask = unique_reverse == file_idx  # Mask out particles with no images in this file stack
            this_file_stack_indices = self.part_stack_idx[this_file_mask]
            this_file_images = mrc.data[this_file_stack_indices]  # Take slices of images for this data set
            this_file_index_list = part_index_list[this_file_mask]  # Particles indices with images in this file

            for j in range(len(this_file_images)):
                idx = this_file_index_list[j]  # This particle index
                self.part_preloaded_image[idx] = this_file_images[j].copy()

    def load_image(self, index):
        image_file_path_idx = self.part_image_file_path_idx[index]
        image_filename = self.image_file_paths[image_file_path_idx]
        if self.part_preloaded_image is not None and len(self.part_preloaded_image) > 0:
            image = self.part_preloaded_image[index]
        else:
            with mrcfile.mmap(image_filename, 'r') as mrc:
                stack_idx = self.part_stack_idx[index]
                if len(mrc.data.shape)>2:
                    image = mrc.data[stack_idx].copy()         
                else:
                    image = mrc.data.copy()


        return image, image_filename

    def __getitem__(self, index):
        image, image_filename = self.load_image(index)
        image = torch.Tensor(image.astype(np.float32))
        og_idx = self.part_og_idx[index]

        rotation = torch.Tensor(self.part_rotation[index])
        translation = torch.Tensor(self.part_translation[index])

        if self.compute_ctf:
            if not self.has_ctf or self.optics_group_ctfs[og_idx] is None:
                ctf = torch.ones_like(image)
            else:
                ctf = torch.Tensor(
                    self.optics_group_ctfs[og_idx](
                        self.optics_group_stats[og_idx]["image_size"],
                        self.optics_group_stats[og_idx]["pixel_size"],
                        torch.Tensor([self.part_defocus[index][0]]),
                        torch.Tensor([self.part_defocus[index][1]]),
                        torch.Tensor([self.part_defocus[index][2]])
                    )
                ).squeeze(0)
            return {
                "image": image,
                "ctf": ctf,
                "rotation": rotation,
                "translation": translation,
                "idx": index,
                "optics_group_idx": og_idx
            }
        else:
            return {
                "image": image,
                "rotation": rotation,
                "translation": translation,
                "idx": index,
                "optics_group_idx": og_idx
            }

    def __len__(self):
        return len(self.part_rotation)

    def get_state_dict(self) -> Dict:
        return {
            "type": "ParticleDataset",
            "version": "0.0.1",
            "image_file_paths": self.image_file_paths,
            "part_rotation": self.part_rotation,
            "part_translation": self.part_translation,
            "part_defocus": self.part_defocus,
            "part_og_idx": self.part_og_idx,
            "part_stack_idx": self.part_stack_idx,
            "part_image_file_path_idx": self.part_image_file_path_idx,
            "part_norm_correction": self.part_norm_correction,
            "part_noise_group_id": self.part_noise_group_id,
            "optics_group_stats": self.optics_group_stats,
        }

    def set_state_dict(self, state_dict):
        if "type" not in state_dict or state_dict["type"] != "ParticleDataset":
            raise TypeError("Input is not an 'ParticleDataset' instance.")

        if "version" not in state_dict:
            raise RuntimeError("ParticleDataset instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            self.initialize(
                image_file_paths=state_dict["image_file_paths"],
                part_rotation=state_dict["part_rotation"],
                part_translation=state_dict["part_translation"],
                part_defocus=state_dict["part_defocus"],
                part_og_idx=state_dict["part_og_idx"],
                part_stack_idx=state_dict["part_stack_idx"],
                part_image_file_path_idx=state_dict["part_image_file_path_idx"],
                part_norm_correction=state_dict["part_norm_correction"],
                part_noise_group_id=state_dict["part_noise_group_id"],
                optics_group_stats=state_dict["optics_group_stats"]
            )
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
