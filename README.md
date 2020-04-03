# Diffraction tomography with a deep image prior

This repository contains python and tensorflow code that implements 3D diffraction tomography under a variety of priors/regularizers, including total variation (TV), positivity, and now, for the first time, the deep image prior (DIP). We call our technique deep prior diffraction tomography (DP-DT), which reparameterizes the reconstruction as the output of an *untrained* 3D convolutional neural network (CNN). We show that DP-DT outperforms traditional regularizers in terms of addressing the missing cone problem, which we demonstrate in a variety of samples and with two popular scattering models: the first Born approximation and the multi-slice model.

For more information on implementation and results, see our arXiv preprint (https://arxiv.org/abs/1912.05330). The peer-reviewed article is currently in press in *Optics Express*.

## Data
We provide the raw multi-angle data stacks for all of our experimental results (Figs. 7-10). These include the 1-um bead sample, the 2-layer 2-um bead sample, and the starfish embryo sample, which can be downloaded from [here](https://doi.org/10.6084/m9.figshare.12081708) as `.mat` files (~150 MB each). The jupyter notebook assumes these files are stored in `DPDT_raw_data/` in the same directory as the python files.

## Code
Run this code using Python 3, with the following libraries installed:
- tensorflow (version 1.x, preferably the GPU version)
- numpy
- scipy
- matplotlib
- jupyter

This code should work with later versions of tensorflow (I've tested 1.14), but not with tensorflow 2, at least in its current state. The optimization gets a significant speedup with a GPU, so I recommend using one if you have one. Note that DP-DT can get memory intensive, as it needs to store both the diffraction tomography forward model and the 3D DIP. Thus, if your GPU doesn't have enough memory, try decreasing the field of view and/or the batch size. The default fields of view and batch sizes in the code can be supported by 16-GB GPU.

## Citation
If you find our code useful to your research, please consider citing the accompanying publication:

TBD
