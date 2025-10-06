import sys
sys.path.append('../modules')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
try:
    from nvitop import Device
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import matplotlib.pyplot as pl
#import sys
#sys.path.append('../modules')
import mlp
#import datasets # L: THIS MODULE (AND ITS CONTENTS ARE NOT HERE! THEY WERE IN THE PREVIOUS CODE THAT WAS SENT)
import dataset # L: modified
import normalize
import symlog
import resnet
import glob
from einops import rearrange

NVIDIA_SMI = False # L: added
    
class CLIPLoss(nn.Module):
    """ Simple contrastive loss for CLIP
    """
    def get_logits(self, z1_features, z2_features, logit_scale):
        logits_per_z1 = logit_scale * z1_features @ z2_features.T
        logits_per_z2 = logit_scale * z2_features @ z1_features.T
        return logits_per_z1, logits_per_z2

    def forward(self, z1_features, z2_features, logit_scale):
        logits_per_z1, logits_per_z2 = self.get_logits(z1_features, z2_features, logit_scale)        
        labels = torch.arange(logits_per_z1.shape[0], device=z1_features.device, dtype=torch.long)
        total_loss = 0.5 * (F.cross_entropy(logits_per_z1, labels) + F.cross_entropy(logits_per_z2, labels))
        return total_loss
    
class Training(object):
    def __init__(self, checkpoint, gpu, batch_size):

        print(f"Loading model {checkpoint}")
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        self.loss = chk['loss']
        self.loss_val = chk['loss_val']

        chk = torch.load(checkpoint+'.best', map_location=lambda storage, loc: storage)

        self.config = chk['config']

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        #if (NVIDIA_SMI):
        #    self.handle = Device.all()[self.gpu]
            
        #    print("Computing in {0} : {1}".format(self.device, self.handle.name()))
        # L: modified this to keep it from crashing again
        if NVIDIA_SMI and torch.cuda.is_available():
            try:
                self.handle = Device.all()[self.gpu]
                print(f"Computing in {self.device} : {self.handle.name()}")
            except Exception:
                print("NVIDIA device not found, running on CPU instead.")
                self.device = torch.device("cpu")
                self.handle = None
            else:
                print("Running on CPU (no GPU or NVML detected).")
                self.device = torch.device("cpu")
                self.handle = None

        self.batch_size = batch_size
        
        # Model
        # self.encoding_pars = mlp.MLP(n_input=9,
        #                             n_output=64,
        #                             dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
        #                             n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
        #                             activation=nn.ReLU()).to(self.device)
        
        # self.encoding_stokes = mlp.MLP(n_input=400,
        #                             n_output=64,
        #                             dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
        #                             n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
        #                             activation=nn.ReLU()).to(self.device)
        
        #self.encoding_models = resnet.ResidualNet(in_features=6*11, 
        #              out_features=self.config['mlp']['latent_dim'],
        #              hidden_features=self.config['mlp']['n_hidden_mlp'],
        #              num_blocks=self.config['mlp']['num_layers_mlp'],
        #              activation=F.gelu,
        #              dropout_probability=0.1,
        #              use_batch_norm=True).to(self.device)
        
        # L: modified to match train_clip.py due to size mismatch error
        self.encoding_models = resnet.ResidualNet(in_features=6*80,
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)
        
        self.encoding_stokes = resnet.ResidualNet(in_features=4*112, 
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)

        print("Setting weights of the model...")        
        #self.encoding_models.load_state_dict(chk['encoding_models_dict'])
        #self.encoding_stokes.load_state_dict(chk['encoding_stokes_dict'])
        # L: modified to match the checkpoints saved in train
        self.encoding_models.load_state_dict(chk['encoder_models_dict'])
        self.encoding_stokes.load_state_dict(chk['encoder_stokes_dict'])

        self.stats = chk['stats']
        
        self.encoding_stokes.eval()
        self.encoding_models.eval()

        self.loss_fn = CLIPLoss()

    def denormalize(self, models):
        models = normalize.denormalize(models, self.stats['min_models'][:,None,None], self.stats['max_models'][:,None,None])
        models[2, ...] = symlog.inv_symlog(models[2, ...])
        models[4:, ...] = symlog.inv_symlog(models[4:, ...])

        models[2, ...] = np.log10(models[2, ...])
        models[3, ...] = models[3, ...] * 1e-5
        return models

    def project(self, array, batch_size, which='stokes'):
        ns, nl, n = array.shape
        array_flat = array.reshape((ns*nl, n)).T
        array_flat = np.array_split(array_flat, batch_size, axis=0)
        z_all = []
        with torch.no_grad():
            for i in tqdm(range(len(array_flat))):
                tmp = torch.tensor(array_flat[i].astype('float32')).to(self.device)
                if (which == 'stokes'):
                    z = self.encoding_stokes(tmp)
                if (which == 'models'):
                    z = self.encoding_models(tmp)
                z = F.normalize(z, dim=-1)
                z_all.append(z.cpu().numpy())
        
        z_all = np.concatenate(z_all, axis=0)

        return z_all    
                
    def test(self):
        
        # L: TEMPORARILY CHANGED TO MATCH INVERT_CLIP.PY 
        #self.training_dataset = datasets.Dataset1D(self.config['training_set'], 
        #            pctx=[0,80], 
        #            pcty=[0,100], 
        #            stats=None, 
        #            step=5)      
        self.training_dataset = dataset.Dataset('stokes_training.h5', 
                                                   'models_training.h5', 
                                                   'good_profiles_training.npy',
                                                   noise=self.config['training']['noise'])          
        
        print("Adding noise to Stokes...")
        stokes_all = []
        for i in range(4):            
            stokes = self.training_dataset.stokes[i] + np.random.normal(loc=0, scale=3e-4, size=self.training_dataset.stokes[i].shape)
            stokes_all.append(stokes.reshape((4, 112, -1)))
        stokes_all = np.concatenate(stokes_all, axis=-1)        
        
        print("Encoding Stokes...")
        z_stokes = self.project(stokes_all, 256, which='stokes')        

        print("Encoding models...")
        models_all = []
        for i in range(4):            
            models = self.training_dataset.models[i][:, ...]
            models_all.append(models.reshape((7, 11, -1)))
        models_all = np.concatenate(models_all, axis=-1)        

        z_models = self.project(models_all[1:, ...], 256, which='models')        

        print("Denormalizing models...")
        models = self.denormalize(models_all)
        
        return z_stokes, z_models, self.loss, self.loss_val, models_all, stokes_all


if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*.pth')
    files.sort()
    checkpoint = files[-1]
    
    deepnet = Training(checkpoint, gpu=0, batch_size=1024*16)
    z_stokes, z_models, loss, loss_val, models, stokes = deepnet.test()

    which = 252000

    z1 = z_stokes[which, :]
    
    sim = np.sum(z1[None, :] * z_models, axis=-1)

    ind = np.argsort(sim)[::-1][0:5]

    labels = ['T', 'logP', 'vz', 'Bp1', 'Bp2', 'Bz']

    fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i in range(6):
        ax.flat[i].plot(models[i, :, which], linewidth=2, color='black')
        ax.flat[i].set_ylabel(labels[i])
        for j in range(5):
            ax.flat[i].plot(models[i, :, ind[j]])

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(4):
        ax.flat[i].plot(stokes[i, :, which], linewidth=2, color='black')
        for j in range(5):
            ax.flat[i].plot(stokes[i, :, ind[j]])