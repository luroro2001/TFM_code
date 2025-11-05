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
import matplotlib
matplotlib.use('QtAgg') # test
import sys
sys.path.append('../modules')
import resnet
import dataset
import normalize
import symlog
import resnet
import glob
from einops import rearrange
import random

def normalize_input(x, xmin, xmax):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def denormalize_output(x, xmin, xmax):
    return 0.5 * (x + 1.0) * (xmax - xmin) + xmin

    
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
    
class Testing(object):
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

        if (NVIDIA_SMI):
            try:
                self.handle = Device.all()[self.gpu]
                print("Computing in {0} : {1}".format(self.device, self.handle.name()))
            except Exception:
                print("NVIDIA device not found, running on CPU instead.")
                self.device = torch.device("cpu")
                self.handle = None
            else:
                print("Running on CPU (no GPU or NVML detected).")
                self.device = torch.device("cpu")
                self.handle = None

        self.batch_size = batch_size
        self.decoders = self.config['training']['use_decoders']
        
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
        
        self.encoder_models = resnet.ResidualNet(in_features=6*80, 
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)
        
        self.encoder_stokes = resnet.ResidualNet(in_features=4*112, 
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)
        
        if self.decoders:
            self.decoder_models = resnet.ResidualNet(in_features=self.config['mlp']['latent_dim'],
                      out_features=6*80,
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)

            self.decoder_stokes = resnet.ResidualNet(in_features=self.config['mlp']['latent_dim'],
                        out_features=4*112,
                        hidden_features=self.config['mlp']['n_hidden_mlp'],
                        num_blocks=self.config['mlp']['num_layers_mlp'],
                        activation=F.gelu,
                        dropout_probability=self.config['mlp']['dropout_probability'],
                        use_batch_norm=True).to(self.device)

        print("Setting weights of the model...")        
        self.encoder_models.load_state_dict(chk['encoder_models_dict'])
        self.encoder_stokes.load_state_dict(chk['encoder_stokes_dict'])

        self.encoder_stokes.eval()
        self.encoder_models.eval()

        if self.decoders:
            self.decoder_models.load_state_dict(chk['decoder_models_dict'])
            self.decoder_stokes.load_state_dict(chk['decoder_stokes_dict'])

            self.decoder_models.eval()
            self.decoder_stokes.eval()
        
    def denormalize(self, models):
        lower = [2000., 0.0, -10.0, 0.0, -1000.0, -1000.0]
        upper = [25000., 3.0, 10.0, 1000.0, 1000.0, 1000.0]
        
        for i in range(models.shape[1]):
            models[:, i, :] = denormalize_output(models[:, i, :], lower[i], upper[i])

        return models
                
    def test(self):

        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}

        self.test_dataset = dataset.Dataset('stokes_testing.h5', 
                                                   'models_testing.h5', 
                                                   'good_profiles_testing.npy',
                                                   noise=self.config['training']['noise'])
        
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False, 
                    **kwargs)
        
        t = tqdm(self.test_loader)

        z_stokes = []
        z_models = []
        models_all = []
        stokes_all = []
        decoded_stokes_all = []
        decoded_models_all = []

        print("Inspecting dataset shapes...")
        for stokes, models in self.test_loader:
            print("Stokes shape:", stokes.shape)
            print("Models shape:", models.shape)
            print("Example values:", models[0, :, :5])  # print first few values per channel
            break

        with torch.no_grad():

            for batch_idx, (stokes, models) in enumerate(t):
                models = models.to(self.device)
                stokes = stokes.to(self.device)

                stokes_flat = rearrange(stokes, 'b c h -> b (c h)')
                models_flat = rearrange(models, 'b c h -> b (c h)')

                z_s = self.encoder_stokes(stokes_flat)
                z_m = self.encoder_models(models_flat)

                z_s = F.normalize(z_s, dim=-1)
                z_m = F.normalize(z_m, dim=-1)

                z_stokes.append(z_s.cpu().numpy())
                z_models.append(z_m.cpu().numpy())

                models_all.append(models.cpu().numpy())
                stokes_all.append(stokes.cpu().numpy())

                if self.decoders:
                    decoded_stokes = self.decoder_stokes(z_s)
                    decoded_models = self.decoder_models(z_m)

                    decoded_stokes = rearrange(decoded_stokes, 'b (c h) -> b c h', c=4)
                    decoded_models = rearrange(decoded_models, 'b (c h) -> b c h', c=6)
                    
                    decoded_stokes_all.append(decoded_stokes.cpu().numpy())
                    decoded_models_all.append(decoded_models.cpu().numpy())

                
        z_stokes = np.concatenate(z_stokes, axis=0)
        z_models = np.concatenate(z_models, axis=0)
        models_all = np.concatenate(models_all, axis=0)
        stokes_all = np.concatenate(stokes_all, axis=0)
        decoded_models_all = np.concatenate(decoded_models_all, axis=0) if self.decoders else None
        decoded_stokes_all = np.concatenate(decoded_stokes_all, axis=0) if self.decoders else None

        models_all = self.denormalize(models_all)
        decoded_models_all = self.denormalize(decoded_models_all) if self.decoders else None

        return z_stokes, z_models, models_all, stokes_all, decoded_models_all, decoded_stokes_all
    
    def plot_reconstruction(self, stokes, decoded_stokes, models, decoded_models, n_samples=3):
        """
        Plot original vs reconstructed Stokes profiles and model parameters
        for a few random test samples.

        Parameters
        ----------
        stokes : np.ndarray, shape (N, 4, 112)
            Original Stokes profiles (I, Q, U, V) for all test samples.
            - N = number of samples
            - 4 = four Stokes parameters
            - 112 = spectral points per profile

        decoded_stokes : np.ndarray, shape (N, 4, 112)
            Reconstructed Stokes profiles predicted by the decoder.

        models : np.ndarray, shape (N, 6, 80)
            Original physical model parameters (temperature, velocity, field components, etc.)
            - 6 = number of physical quantities
            - 80 = depth or height grid points in the solar atmosphere

        decoded_models : np.ndarray, shape (N, 6, 80)
            Reconstructed model parameters from the decoder.

        n_samples : int
            Number of random examples to visualize.
        """

        # NORMALIZATION
        #lower_stokesI, upper_stokesI = 0.0, 2.5
        #lower_stokesQ, upper_stokesQ = -1e-2, 1e-2
        #lower_stokesU, upper_stokesU = -1e-2, 1e-2
        #lower_stokesV, upper_stokesV = -1e-2, 1e-2
        #lower_T, upper_T = 2000, 25000
        #lower_vmic, upper_vmic = 0.0, 3.0
        #lower_v, upper_v = -10.0, 10.0
        #lower_Bx, upper_Bx = 0.0, 1000.0
        #lower_By, upper_By = -1000.0, 1000.0
        #lower_Bz, upper_Bz = -1000.0, 1000.0

        # --- Normalize Stokes ---
        #I = stokes[:, 0, :]
        #Q = stokes[:, 1, :] / I
        #U = stokes[:, 2, :] / I
        #V = stokes[:, 3, :] / I
        #stokes_norm = np.stack([normalize_input(I, lower_stokesI, upper_stokesI),
        #    normalize_input(Q, lower_stokesQ, upper_stokesQ),
        #    normalize_input(U, lower_stokesU, upper_stokesU),
        #    normalize_input(V, lower_stokesV, upper_stokesV)
        #], axis=1)

        # --- Normalize models ---
        #models_norm = np.stack([
        #    normalize_input(models[:, 0, :], lower_T, upper_T),
        #    normalize_input(models[:, 1, :], lower_vmic, upper_vmic),
        #    normalize_input(models[:, 2, :], lower_v, upper_v),
        #    normalize_input(models[:, 3, :], lower_Bx, upper_Bx),
        #    normalize_input(models[:, 4, :], lower_By, upper_By),
        #    normalize_input(models[:, 5, :], lower_Bz, upper_Bz)
        #], axis=1)

        #decoded_stokes_norm = decoded_stokes
        #decoded_models_norm = decoded_models

        n_total = stokes.shape[0]
        indices = random.sample(range(n_total), min(n_samples, n_total))

        stokes_labels = ["I", "Q", "U", "V"]
        #model_labels = ["T", "log_tau", "v", "Bz", "Bx", "By"] # WROOOOOOOOOOOOONG
        model_labels = ['T', 'vmic', 'v', 'Bx', 'By', 'Bz']

        for idx in indices:
            fig, axes = pl.subplots(2, 1, figsize=(10, 8))
            #fig.suptitle(f"Sample {idx}", fontsize=14, fontweight="bold")

            # --- Stokes profiles ---
            ax = axes[0]
            for s in range(4):
                ax.plot(stokes[idx, s], label=f"{stokes_labels[s]}")
                ax.plot(decoded_stokes[idx, s], "--", label=f"{stokes_labels[s]} (decoded)")
            ax.set_title("Stokes profiles")
            ax.set_xlabel("Wavelength index")
            ax.set_ylabel("Normalized intensity")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # --- Model parameters ---
            ax = axes[1]
            for m in range(6):
                ax.plot(models[idx, m], label=f"{model_labels[m]}")
                ax.plot(decoded_models[idx, m], "--", label=f"{model_labels[m]} (decoded)")
            ax.set_title("Physical model parameters")
            ax.set_xlabel("Depth index")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8, ncol=3)
            #ax.grid(alpha=0.3)

            pl.tight_layout(rect=[0, 0, 1, 0.96])
            pl.show()
            output_file = f"reconstruction_sample_{idx}.png"
            #pl.savefig(output_file, dpi=150)
            print(f"Saved {output_file}")
            pl.close()



if (__name__ == '__main__'):

    #files = glob.glob('../train/weights/*.pth')
    #files.sort()

    checkpoint = '../train/weights/2025-10-05-20_58_50_clip.pth'
    deepnet = Testing(checkpoint, gpu=0, batch_size=4)
    z_stokes, z_models, models, stokes, decoded_models, decoded_stokes = deepnet.test()
    deepnet.plot_reconstruction(stokes, decoded_stokes, models, decoded_models, n_samples=3)
