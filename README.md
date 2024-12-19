# DynaMight

[![License](https://img.shields.io/pypi/l/dynamight.svg?color=green)](https://github.com/schwabjohannes/dynamight/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/dynamight.svg?color=green)](https://pypi.org/project/dynamight)
[![Python Version](https://img.shields.io/pypi/pyversions/dynamight.svg?color=green)](https://python.org)
[![CI](https://github.com/schwabjohannes/dynamight/actions/workflows/ci.yml/badge.svg)](https://github.com/schwabjohannes/dynamight/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/schwabjohannes/dynamight/branch/main/graph/badge.svg)](https://codecov.io/gh/schwabjohannes/dynamight)

Estimating dynamics from cryo-EM images and use them to improve your map (maybe)

Analysing continuous heterogeneity in a cryo-EM dataset with dynamight consists of 3 main steps:

## Installation

We recommend to install DynaMight with RELION. For a detailed installation intruction see (https://relion.readthedocs.io/en/release-5.0/Installation.html)

*DynaMight* is available from the Python package index ([PyPI](https://pypi.org/)).

Please install using `pip` in a clean virtual environment.

```shell
pip install dynamight
```

## Step 1: Estimation and inspection of deformations

### Training the VAEs

To start the estimation of fexibility in a cryo EM dataset it is necessary to have a .star file from a previously done consensus refinement. The dynamight function that trains the VAE is **optimize deformations** and its minimum requirement is the .star file and the name of an output directory. Although there is an option for not using a consensus map for initialization it is recommended to use the map from the consensus refinement as initial model. With that at hand, you can already run the VAE training by

**dynamight optimize-deformations run_data.star /dataset/output --initial-model map.mrc**

There are many more parameters that define the neural network that in normal usecases don't have to be changed. One important parameter that defines the resolution of the Gaussian model is the number of gaussians, which can be changed by the **--n-gaussians**. Other arguments that might be changed are the number of latent dimensions **--n-latent-dims** and the strenght of regularization **--regularization-factor**. For a regularization factor of 1 the regularization parameter is calculated, such the the norm of gradients from the regularization is equal to the norm of the gradients coming from the data-fidelity term. A smaller parameter gives more weight to the data, whereas a larger parameter implies stronger regularization. There is the option to provide a mask file, which sets the deformations outside of the mask to zero. This can be done by adding **--mask-file mask.mrc**. 

It is also possible to follow the training process by looking at the tensorboard that is updated after every epoch.

### Training the VAE with a rigid body prior

A modified version of Dynamight estimates rigid deformations for user defined bodies. In this version, the Encoder is exactly the same, whereas the decoder predicts a rotation and translation for each body that is specified by binary masks (--deformation-masks). For a given latent representation $z$ and $N$ masks $\{m_1,\ldots, m_N\}$ is of the form 

$$ D_z(x) = \sum_{i=1}^N \sum_{j=1}^{N_i} \varphi_j(R_{i,z}(x)),$$

where $N_i$ is the number of Gaussians within mask $m_i$ and $R_{i,z}§ is the rigid transform that is estimated for mask $m_i$ from the latent representation $z$. 


### Visualization 

Interactive visualization can be done using the function **explore-latent-space**. The input to this function is the path to the output directory of the training job, and a checkpoint file that stores the weights and metadata from the training run.

**dynamight explore-latent-space /dataset/output --checkpoint-file /dataset/output/forward_deformations/checkpoints/120.pth**

To select the half set that is visualized, you can use **--half-set** to select 1,2 or 0. The default is 1 and if 0 is selected the deformations for the validation set is used, which also shows per particle error estimates of the deformations. Once the latent space and dimensionality reduction is done, a napari window should open that looks like this:

![](https://github.com/schwabjohannes/DynaMight/blob/main/napari.png)

On the right hand side of the viewer the latent space is displayed, where every point corresponds to one particle image. By clicking in latent space the map on the left side is updated and shows the corresponding conformation. Below the visualization of latent space you can choose the property by which latent space is colored (e.g. direction of movement, amount, ... ). On the left panel you can also visualize the consensus structure or a point cloud representation of the gaussian model. Changing the dropdown menu **3d representation** to points will update the point-cloud after clicking in latent space. Additionally to the described click functionality you can choose other applications in the action dropdown menu. You can for example choose trajectory to draw a curve and get a movie of the conformational states along the curve, determine the particle number in a region of latent space or write out a starfile containing all particles inside a region. You can save volume series or single maps by clicking the button **save map/movie**.

## Step 2: Computing the inverse deformations

To use the estimated deformations in a modified weighted backprojection. We have to estimate the inverse deformations to map all the original particles back to a consensus conformation. To do this you have to run the program **optimize-inverse-deformations**, which takes the path to the ouput directory of step 1 and the vae checkpoint as an input.

**dynamight optimize-inverse-deformations /dataset/output --checkpoint-file /dataset/output/forward_deformations/checkpoints/120.pth**

Additional parameters are the number of epochs (**--n-epochs**) and the possibility to store the deformations in RAM (**--save-deformations**), which speeds up the training process.

## Step 3: Deformed backprojection

Once the inverse deformations have been estimated, they can be used for a backprojection algorithm to improve the reconstruction in felxible areas. The backprojection is implemented in the function **deformable-backprojection**, which takes the checkpoint file of the forward deformations as input.

**dynamight deformable-backprojection /dataset/output --vae-directory /dataset/output/forward_deformations/checkpoints/120.pth**

This function also has two additional useful arguments. The first one is **--downsample**, which takes an integer to decrease the computation of the inverse deformation field to a smaller box and then upsamples it to the full size. This makes it computationally more efficient, using less network evaluations. The second argument is **--backprojection-batch-size** that has to be adapted to the memory size of the gpu.
