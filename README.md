# Project structure

TFM/code/
│ 
├── database 
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
│   ├── dataset.py
│   ├── encoder_decoder.py
│   ├── encoding.py
│   ├── mlp.py
│   ├── normalize.py
│   ├── resnet.py
│   ├── siren.py
│   └── symlog.py
│   
├── train 
│   ├── weights
│   ├── train_clip.py
│   └── train_vicreg.py
│ 
├── validate
│   ├── validate.py
│   └── validate_trial.py
│ 
├── validate_old
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

In the following sections, each script that makes up the code will be explained.

# **`database`**

**PENDING**: add explanation of what each file is.

# **`modules`**

## **`dataset.py`**

It starts off by defining two helper functions to normalize and denormalize data. 

**normalize_input** scales input data $x$ from the range $[xmin, xmax]$ to $[-1, 1]$. This is because neural networks work better when the inputs are normalized. The formula is simply:

$x_{norm}=2 \cdot \frac{x-x_{min}}{x_{max}-x_{min}} - 1$

**denormalize_input** reverts normalized values from $[-1, 1]$ back to the original range $[xmin, xmax]$.

### **`Dataset class`**

Provides both Stokes profiles abd physical model parameters for training.

Its inputs are:
- `filename_stokes`: HDF5 file containing Stokes I, Q, U, V profiles.
- `filename_model`: HDF5 file containing the physical model parameters (logtau, T, Pe, vmic, v, Bx, By, Bz).
- `good_profiles_filename`: .npy file indexing "good" profiles to use.
- `n_training`: optional, number of examples to train on.
- `noise`: Amount of gaussian noise to add for augmentation. 