# Foundation Model for Solar Spectropolarimetry

This master's thesis (TFM) presents the development and evaluation of a foundational model for solar spectropolarimetry employing contrastive training. The model consists of two residual network encoders, one for Stokes profiles and one for physical atmospheric models, trained together to project both modalities into a shared latent space using a CLIP style contrastive loss, as well as reconstruction losses from the two corresponding decoders. The training database consists of synthetic profiles of the Fe I doublet at 630.15 and 630.25 nm, computed from perturbations of semi-empirical solar atmospheric models, covering a wide range of physical conditions that are representative of different solar regions.

<img width="1541" height="893" alt="Image" src="https://github.com/user-attachments/assets/afd27cf2-10f9-42d6-99d6-df799ef9c85c" />

---

## Table of contents

- [Project Structure](#project-structure)
- [How Does It Work?](#how-does-it-work)
- [Downstream Tasks](#downstream-tasks)
- [Requirements](#requirements)

## Project structure

```
TFM/code/
│ 
├── database # NOT INCLUDED IN THIS REPOSITORY
│   ├── good_profiles_testing.npy
│   ├── good_profiles_training.npy
│   ├── good_profiles_validation.npy
│   ├── models_testing.h5
│   ├── models_training.h5
│   ├── models_validation.h5
│   ├── stokes_testing.h5
│   ├── stokes_training.h5
│   └── stokes_validation.h5
│ 
├── modules 
│   ├── dataset.py # to load the data
│   ├── encoder_decoder.py # currently unused
│   ├── encoding.py # currently unused
│   ├── mlp.py # currently unused
│   ├── normalize.py # normalization and denormalization fucntions
│   ├── resnet.py # defines the residual networks
│   ├── siren.py # currently unused
│   └── symlog.py # symmetric logarithm transform
│   
├── train 
│   ├── weights/ # saved model checkpoints. NOT INCLUDED/UPDATED HERE
│   ├── train_clip.py # CLIP style contrastive training
│   ├── train_vicreg.py # VICReg style training. 
│   └── conf.yaml # hyperparameters
│ 
├── validate
│   ├── validate.py #validation script
│   └── validate_trial.py
│ 
├── validate_old # old scripts, UNUSED
│   ├── deconvolution_hinode.py
│   ├── doplots_clip.py
│   ├── invert_clip.py
│   ├── invert_vicreg.py
│   ├── invert.py
│   ├── noise_svd.py
│   ├── validate_2d.py
│   ├── validate.py
│   ├── view_models.py
│   └── view.py
│
└── README.md
```
---

## How does it work?

EXPLAIN EVERYTHING HERE, TAKE FROM TFM.

---

## Downstream tasks

The model currently allows the execution of two downstream applications:

### Fast Stokes inversion

The fast Stokes inverter implements the cross-modal path that is illustrated in the figure below. Given a Stokes profiles as input, the Stokes encoder projects it into the shared latent space, producing a latent vector $\textbf{z}_s \in \mathbb{R}^{64}$. This vector is then decoded by the model decoder to produce an estimation of the atmospheric stratification for each of the six physical parameters $T$, $v$, $v_{\rm mic}$, $B_x$, $B_y$ and $B_z$. The main advantage of the approach is that it avoids the iterative nature of classical inversion codes; once the model is trained, the inversion of a profile only requires two passes through two neural networks, so it can be carried out faster than in traditional methods.

<img width="2718" height="841" alt="Image" src="https://github.com/user-attachments/assets/47cb75c9-ad89-4e79-b9f7-e515bb6d8294" />

### Fast Stokes synthesis

The fast Stokes synthesizer implements the cross-modal path that is illustrated in the figure below. Given a physical atmospheric model as input, the model encoder projects it into the shared latent space, producing a latent vector $\textbf{z}_m$ $\in$ $\mathbb{R}^{64}$, which is then decoded by the Stokes decoder to produce a synthetic Stokes profile. As with the inverter, this path was not part of the explicit training, so its performance reflects the quality of the contrastive alignment between the two encoders. The synthesizer represents the forward problem: given known physical conditions, producing the corresponding observational parameters.

<img width="2716" height="788" alt="Image" src="https://github.com/user-attachments/assets/f39e6fec-7bd2-4b64-9e0c-eee9c984e5d9" />

---

## Requirements

The model was implemented in Python using the following packages:

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch](https://pytorch.org/)
- [h5py](https://www.h5py.org/)
- [scikit-learn](https://scikit-learn.org/)
