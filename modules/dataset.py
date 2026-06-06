import torch
import torch.utils.data
import h5py
import numpy as np
import matplotlib.pyplot as pl

# L: scales input data x from range [xmin, xmax] to [-1,1]
# L: nn's train better when inputs are normalized, so this is applied to all parameters 
# before the are fed into the network 
def normalize_input(x, xmin, xmax):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0 #L: linear scalng formula


# L: reverts normalized values from [-1,1] back to [xmin, xmax]
# L: defined again at the start of validate.py
def denormalize_output(x, xmin, xmax):
    return 0.5 * (x + 1.0) * (xmax - xmin) + xmin # L: despejando x de la funcion anterior

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for the dataset. This one shows how to do augmenting (L: ?, leftover comment) during training for a 
    for a very simple training set 

    L: This class provides both stokes profiles and physical model parameters for training.   
    """
    def __init__(self, filename_stokes, filename_model, good_profiles_filename, n_training=None, noise=0.0):
        """
        Initialize the dataset (synthetic Stokes and their atmospheric modles) stores in HDF5 files.
        
        Args:
            n_training (int): number of training examples including augmenting
            filename_stokes: HDF5 file with Stokes I, Q, U, V profiles
            filename_model: HDF5 file with the physical model parameters (temp, vel, B_i, etc)
            good_profiles_filename: numpy files with indices of valid profiles to use
            noise: stdv of gaussian noise added to the Stokes profiles
        """
        super(Dataset, self).__init__()

        self.noise = noise

        # L: opens the hdf5 files containing the data
        f_stokes = h5py.File(f'../database/{filename_stokes}', 'r')
        f_model = h5py.File(f'../database/{filename_model}', 'r')

        # L: loads indices of good profiles (filtering purposes)
        ind = np.load(f'../database/{good_profiles_filename}')

        # Models contain the following parameters:
        # logtau, T, Pe, vmic, v, Bx, By, Bz
        print("Reading Stokes profiles and models from file...")
        self.stokes = f_stokes['spec1']['stokes'][:]
        self.model = f_model['model'][:]

        print("Selecting good profiles...")
        self.model = self.model[ind, ...]
        self.stokes = self.stokes[ind, ...]

        self.model = np.transpose(self.model, (0, 2, 1)) # L:transpose so that dimensions match
        
        # L: sets dataset length: either all available samples or a subset
        if n_training is None:
            self.n_training = self.stokes.shape[0]
        else:
            self.n_training = n_training

        # L: normalization bounds (min and max values?). Later used in normalize_input()        
        #self.lower_stokesI = 0.0
        #self.upper_stokesI = 2.5
        self.lower_stokesI = 0.8
        self.upper_stokesI = 1.2

        self.lower_stokesQ = -1e-2
        self.upper_stokesQ = 1e-2

        self.lower_stokesU = -1e-2
        self.upper_stokesU = 1e-2

        #self.lower_stokesV = -1e-2
        #self.upper_stokesV = 1e-2

        self.lower_stokesV = -5e-2
        self.upper_stokesV = 5e-2

        self.lower_T = 2000
        self.upper_T = 25000

        # L: 'vmic' is microturbulent velocity
        self.lower_vmic = 0.0
        self.upper_vmic = 3.0

        self.lower_v = -10.0
        self.upper_v = 10.0

        self.lower_Bx = 0.0
        self.upper_Bx = 1000.0

        self.lower_By = -1000.0
        self.upper_By = 1000.0

        self.lower_Bz = -1000.0
        self.upper_Bz = 1000.0
                
    def __getitem__(self, index): # L: called for each sample during training

        # Add noise and normalize Stokes QUV by Stokes I
        #(L: converts polarization signals into fractional polarizaion (Q/I, U/I, V/I), standard in spectropolarimetry)
        out_stokesI = self.stokes[index, 0, 0, :] # L: gets Stokes I for the given index
        if self.noise != 0:
            out_stokesI += np.random.normal(0, self.noise, out_stokesI.shape)

        out_stokesQ = self.stokes[index, 0, 1, :]
        if self.noise != 0:
            out_stokesQ += np.random.normal(0, self.noise, out_stokesQ.shape)
        out_stokesQ /= out_stokesI

        out_stokesU = self.stokes[index, 0, 2, :]
        if self.noise != 0:
            out_stokesU += np.random.normal(0, self.noise, out_stokesU.shape)
        out_stokesU /= out_stokesI

        out_stokesV = self.stokes[index, 0, 3, :]
        if self.noise != 0:
            out_stokesV += np.random.normal(0, self.noise, out_stokesV.shape)
        out_stokesV /= out_stokesI

        # L: rescales all inputs into [-1, 1] for neural network stability
        out_stokesI = normalize_input(out_stokesI, self.lower_stokesI, self.upper_stokesI)        
        out_stokesQ = normalize_input(out_stokesQ, self.lower_stokesQ, self.upper_stokesQ)
        out_stokesU = normalize_input(out_stokesU, self.lower_stokesU, self.upper_stokesU)
        out_stokesV = normalize_input(out_stokesV, self.lower_stokesV, self.upper_stokesV)

        # L: Repeats for temperature (T), microturbulent velocity (vmic), 
        # LOS velocity (v), magnetic field components (Bx, By, Bz). 
        out_T = self.model[index, 1, :]
        out_T = normalize_input(out_T, self.lower_T, self.upper_T)

        out_vmic = self.model[index, 3, :]
        out_vmic = normalize_input(out_vmic, self.lower_vmic, self.upper_vmic)

        out_v = self.model[index, 4, :]
        out_v = normalize_input(out_v, self.lower_v, self.upper_v)

        out_Bx = self.model[index, 5, :]
        out_By = self.model[index, 6, :]
        out_Bz = self.model[index, 7, :]
        out_Bx = normalize_input(out_Bx, self.lower_Bx, self.upper_Bx)
        out_By = normalize_input(out_By, self.lower_By, self.upper_By)
        out_Bz = normalize_input(out_Bz, self.lower_Bz, self.upper_Bz)

        out_stokes = np.concatenate((out_stokesI[None, :], out_stokesQ[None, :], out_stokesU[None, :], out_stokesV[None, :]), axis=0)
        out_model = np.concatenate((out_T[None, :], out_vmic[None, :], out_v[None, :], out_Bx[None, :], out_By[None, :], out_Bz[None, :]), axis=0)

        # L: (the correct pairings are maintained because of the common indices across both datasets)
        return out_stokes.astype('float32'), out_model.astype('float32')

    def __len__(self):
        return self.n_training # L:returns number of samples in the dataset
    

class DatasetHinode(torch.utils.data.Dataset):
    """
    Dataset class that could provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set. L: LEFTOVER OLD COMMENT, this is not actually used.

    L: similar to Dataset but tailored for real Hinode solar data (no physical model parameters).
    It's not used in the model.
    """
    def __init__(self, filename_stokes, startx=0, starty=0, nx=0, ny=0):
        """
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(DatasetHinode, self).__init__()
        
        f_stokes = h5py.File(f'{filename_stokes}', 'r')
        
        print("Reading Stokes profiles and models from file...")
        if nx == 0 or ny == 0:
            nx, ny = f_stokes['stokes'].shape[1:3]

        # L: extracts subsets from 2D solar images
        self.stokes = f_stokes['stokes'][:, startx:startx+nx, starty:starty+ny, :]
        
        x = np.arange(nx)
        y = np.arange(ny)
        self.indx, self.indy = np.meshgrid(x, y, indexing='ij')
        self.indx = self.indx.flatten()
        self.indy = self.indy.flatten()
                
        self.n_training = len(self.indx)
                        
        self.lower_stokesI = 0.0
        self.upper_stokesI = 2.5

        self.lower_stokesQ = -1e-2
        self.upper_stokesQ = 1e-2

        self.lower_stokesU = -1e-2
        self.upper_stokesU = 1e-2

        self.lower_stokesV = -1e-2
        self.upper_stokesV = 1e-2

        self.cont = np.mean(f_stokes['stokes'][0, 0:100, 350:, 0])
                
    def __getitem__(self, index):

        indx = self.indx[index]
        indy = self.indy[index]

        # Normalize Stokes I by the continuum and compute Stokes QUV divided by Stokes I (we don't need to normalize by the continuum here)
        out_stokesI = self.stokes[0, indx, indy, :] / self.cont
        out_stokesQ = self.stokes[1, indx, indy, :] / self.stokes[0, indx, indy, :]
        out_stokesU = self.stokes[2, indx, indy, :] / self.stokes[0, indx, indy, :]
        out_stokesV = self.stokes[3, indx, indy, :] / self.stokes[0, indx, indy, :]

        out_stokesI = normalize_input(out_stokesI, self.lower_stokesI, self.upper_stokesI)
        out_stokesQ = normalize_input(out_stokesQ, self.lower_stokesQ, self.upper_stokesQ)
        out_stokesU = normalize_input(out_stokesU, self.lower_stokesU, self.upper_stokesU)
        out_stokesV = normalize_input(out_stokesV, self.lower_stokesV, self.upper_stokesV)
        
        return out_stokesI[None, :].astype('float32'), out_stokesQ[None, :].astype('float32'), out_stokesU[None, :].astype('float32'), out_stokesV[None, :].astype('float32')

    def __len__(self):
        return self.n_training


    
if __name__ == "__main__":
    dataset = Dataset('stokes_training.h5', 'models_training.h5', 'good_profiles_training.npy', n_training=1000, noise=0.0)
    # L: prints first sample and its size to see if the dataset has loaded correctly, Ig.
    #print(dataset[0])
    #print(len(dataset))

    #dataset_test = Dataset('stokes_testing.h5', 'models_testing.h5','good_profiles_testing.npy', n_training=None, noise=0.0)
    
    #snr_indices = select_by_snr(dataset_test, n_per_category=10)
    #print(snr_indices)

    #all_indices = snr_indices['high'] + snr_indices['mid'] + snr_indices['low']
    #plot_noise_effect(dataset_test, all_indices, noise_level=1e-3)