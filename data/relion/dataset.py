#!/usr/bin/env python

"""
Module for loading RELION particle datasets
"""

import os

from glob import glob
import numpy as np

from ..base.particle_dataset import ParticleDataset
from ..base.star_file import load_star


class RelionDataset:
    def __init__(self, path: str = None):
        self.project_root = None
        self.data_star_path = None
        self.preload = None
        self.image_file_paths = []

        # In data star file
        self.part_rotation = []
        self.part_translation = []
        self.part_defocus = []
        self.part_og_idx = []
        self.part_stack_idx = []
        self.part_image_file_path_idx = []
        self.part_norm_correction = []
        self.part_noise_group_id = []
        self.nr_particles = None

        # In data star file
        self.optics_groups = []
        self.optics_groups_ids = []

        if path is not None:
            self.load(path)

    def load(self, path: str) -> None:
        """
        Load data from path
        :param path: relion job directory or data file
        """
        if os.path.isfile(path):
            data_star_path = path
            root_search_path = os.path.dirname(os.path.abspath(path))
        else:
            data_star_path = os.path.abspath(self._find_star_file_in_path(path, "data"))
            root_search_path = os.path.abspath(path)

        self.data_star_path = os.path.abspath(data_star_path)
        data = load_star(self.data_star_path)

        if 'optics' not in data:
            raise RuntimeError("Optics groups table not found in data star file")
        if 'particles' not in data:
            raise RuntimeError("Particles table not found in data star file")

        self._load_optics_group(data['optics'])
        self._load_particles(data['particles'])

        self.project_root = self._find_project_root(root_search_path, self.image_file_paths[0])

        # Convert image paths to absolute paths
        for i in range(len(self.image_file_paths)):
            self.image_file_paths[i] = os.path.abspath(
                os.path.join(self.project_root, self.image_file_paths[i]))

        # TODO check cross reference integrity, e.g. all part_noise_group_id exist in noise_group_id

    def make_particle_dataset(self):
        dataset = ParticleDataset()
        dataset.initialize(
            self.image_file_paths,
            self.part_rotation,
            self.part_translation,
            self.part_defocus,
            self.part_og_idx,
            self.part_stack_idx,
            self.part_image_file_path_idx,
            self.part_norm_correction,
            self.part_noise_group_id,
            self.optics_groups
        )
        return dataset

    def _load_optics_group(self, optics: dict) -> None:
        if 'rlnOpticsGroup' not in optics:
            raise RuntimeError(
                "Optics group id (rlnOpticsGroup) is required, "
                "but was not found in optics group table."
            )

        if 'rlnImageSize' not in optics:
            raise RuntimeError(
                "Image size (rlnImageSize) is required, "
                "but was not found in optics group table."
            )

        if 'rlnImagePixelSize' not in optics:
            raise RuntimeError(
                "Image pixel size (rlnImagePixelSize) is required, "
                "but was not found in optics group table."
            )

        nr_optics = len(optics['rlnOpticsGroup'])

        for i in range(nr_optics):
            id = int(optics['rlnOpticsGroup'][i])
            image_size = int(optics['rlnImageSize'][i])
            pixel_size = float(optics['rlnImagePixelSize'][i])

            if image_size <= 0 or image_size % 2 != 0:
                raise RuntimeError(
                    f"Invalid value ({image_size}) for image size of optics group {id}.\n"
                    f"Image size must be even and larger than 0."
                )
            if pixel_size <= 0:
                raise RuntimeError(
                    f"Invalid value ({pixel_size}) for pixel size of optics group {id}."
                )

            voltage = float(optics['rlnVoltage'][i]) \
                if 'rlnVoltage' in optics else None
            spherical_aberration = float(optics['rlnSphericalAberration'][i]) \
                if 'rlnSphericalAberration' in optics else None
            amplitude_contrast = float(optics['rlnAmplitudeContrast'][i]) \
                if 'rlnAmplitudeContrast' in optics else None

            self.optics_groups_ids.append(id)
            self.optics_groups.append({
                "id": id,
                "image_size": image_size,
                "pixel_size": pixel_size,
                "voltage": voltage,
                "spherical_aberration": spherical_aberration,
                "amplitude_contrast": amplitude_contrast
            })

    def _load_particles(self, particles: dict) -> None:
        if 'rlnImageName' not in particles:
            raise RuntimeError(
                "Image name (rlnImageName) is required, "
                "but was not found in particles table."
            )

        if 'rlnOpticsGroup' not in particles:
            raise RuntimeError(
                "Optics group id (rlnOpticsGroup) is required, "
                "but was not found in particles table."
            )

        nr_particles = len(particles['rlnImageName'])

        for i in range(nr_particles):

            # Optics group ---------------------------------------
            og_id = int(particles['rlnOpticsGroup'][i])
            og_idx = self.optics_groups_ids.index(og_id)
            self.part_og_idx.append(og_idx)
            og = self.optics_groups[og_idx]

            # Norm correction -------------------------------------
            if 'rlnNormCorrection' in particles:
                nc = float(particles['rlnNormCorrection'][i])
                self.part_norm_correction.append(nc)
            else:
                self.part_norm_correction.append(1.)

            # Noise group -----------------------------------------
            if 'rlnGroupNumber' in particles:
                ng = int(particles['rlnGroupNumber'][i])
                self.part_noise_group_id.append(ng)
            else:
                self.part_noise_group_id.append(None)

            # CTF parameters -------------------------------------
            if 'rlnDefocusU' in particles and \
                'rlnDefocusV' in particles and \
                'rlnDefocusAngle' in particles:
                ctf_u = float(particles['rlnDefocusU'][i])
                ctf_v = float(particles['rlnDefocusV'][i])
                ctf_a = float(particles['rlnDefocusAngle'][i])
                self.part_defocus.append([ctf_u, ctf_v, ctf_a])
            else:
                self.part_defocus.append(None)

            # Rotation parameters --------------------------------
            if 'rlnAngleRot' in particles and \
                'rlnAngleTilt' in particles and \
                'rlnAnglePsi' in particles:
                a = np.array([
                    float(particles['rlnAngleRot'][i]),
                    float(particles['rlnAngleTilt'][i]),
                    float(particles['rlnAnglePsi'][i])
                ])
                a *= np.pi / 180.
                self.part_rotation.append(a)
            elif 'rlnAnglePsi' in particles:
                a = np.array([0., 0., float(particles['rlnAnglePsi'][i])])
                a *= np.pi / 180.
                self.part_rotation.append(a)
            else:
                self.part_rotation.append(np.zeros([3]))

            # Translation parameters ------------------------------
            if 'rlnOriginXAngst' in particles and 'rlnOriginYAngst' in particles:
                trans_x = float(particles['rlnOriginXAngst'][i]) / og['pixel_size']
                trans_y = float(particles['rlnOriginYAngst'][i]) / og['pixel_size']
            else:
                trans_x = 0.
                trans_y = 0.
            self.part_translation.append([trans_x, trans_y])

            # Image data ------------------------------------------
            img_name = particles['rlnImageName'][i]
            img_tokens = img_name.split("@")
            if len(img_tokens) == 2:
                image_stack_id = int(img_tokens[0]) - 1
                img_path = img_tokens[1]
            elif len(img_tokens) == 1:
                image_stack_id = 0
                img_path = img_tokens[1]
            else:
                raise RuntimeError(f"Invalid image file name (rlnImageName): {img_name}")

            self.part_stack_idx.append(image_stack_id)

            try:  # Assume image file path has been added to list
                img_path_idx = self.image_file_paths.index(img_path)
                self.part_image_file_path_idx.append(img_path_idx)
            except ValueError:  # If image file path not found in existing list
                img_path_idx = len(self.image_file_paths)
                self.part_image_file_path_idx.append(img_path_idx)
                self.image_file_paths.append(img_path)

        self.part_og_idx = np.array(self.part_og_idx)
        self.part_defocus = np.array(self.part_defocus, dtype=np.float32)
        self.part_rotation = np.array(self.part_rotation, dtype=np.float32)
        self.part_translation = np.array(self.part_translation, dtype=np.float32)
        self.part_noise_group_id = np.array(self.part_noise_group_id)
        self.part_stack_idx = np.array(self.part_stack_idx)
        self.part_image_file_path_idx = np.array(self.part_image_file_path_idx)
        self.nr_particles = len(self.part_image_file_path_idx)

    @staticmethod
    def _find_star_file_in_path(path: str, type: str = "optimiser") -> str:
        if os.path.isfile(os.path.join(path, f"run_{type}.star")):
            return os.path.join(path, f"run_{type}.star")
        files = glob(os.path.join(path, f"*{type}.star"))
        if len(files) > 0:
            files = list.sort(files)
            return files[-1]

        raise FileNotFoundError(f"Could not find '{type}' star-file in path: {path}")

    @staticmethod
    def _find_project_root(from_path: str, file_relative_path: str) -> str:
        """
        Searches for the Relion project root starting at from_path and iterate through parent directories
        till file_relative_path is found as a relative sub path or till filesystem root is found, at which
        point a RuntimeException is raise.

        :param from_path: starting search from this path
        :param file_relative_path: searching to find this relative path as a file
        """
        current_path = os.path.abspath(from_path)
        while True:
            trial_path = os.path.join(current_path, file_relative_path)
            if os.path.isfile(trial_path):
                return current_path
            if current_path == os.path.dirname(current_path):  # At filesystem root
                raise RuntimeError(
                    f"Relion project directory could not be found from the subdirectory: {from_path}")
            current_path = os.path.dirname(current_path)
