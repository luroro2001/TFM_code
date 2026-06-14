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
import sys
import os
#sys.path.append('../modules')
# to fix issues with imports:
MODULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modules')
sys.path.append(MODULES_PATH)
print(f"Added to sys.path: {MODULES_PATH}")
import resnet
import dataset
import normalize
import symlog
import glob
from einops import rearrange
import random 
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import h5py 

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
    # Loads a trained modeland evaluates it on the test dataset
    def __init__(self, checkpoint, gpu, batch_size):

        # Load training data and best checkpoint
        print(f"Loading model {checkpoint}")
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        # Store the training and validation loss histories
        self.loss = chk['loss']
        self.loss_val = chk['loss_val']

        chk = torch.load(checkpoint+'.best', map_location=lambda storage, loc: storage) # best model

        self.config = chk['config'] # load configuration used to build the architecture

        # Check if there is a GPU available and define the computing device (CPU or GPU)
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        # If NVITOP is installed, use it to monitor GPU usage
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

        # Define the batch size and if decoders are used
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
        
        # Define the nueral networks (encoders and decoders)
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
        # Load training weights       
        self.encoder_models.load_state_dict(chk['encoder_models_dict'])
        self.encoder_stokes.load_state_dict(chk['encoder_stokes_dict'])

        self.encoder_stokes.eval()
        self.encoder_models.eval()

        if self.decoders:
            self.decoder_models.load_state_dict(chk['decoder_models_dict'])
            self.decoder_stokes.load_state_dict(chk['decoder_stokes_dict'])

            #L: also put decoders in evaluation mode for plot_reconstruction
            self.decoder_models.eval()
            self.decoder_stokes.eval()

        # Denormalization bounds for model parameters (added for plots)
        self.model_lower = [2000., 0.0, -10.0, 0.0, -1000.0, -1000.0]
        self.model_upper = [25000., 3.0, 10.0, 1000.0, 1000.0, 1000.0]
        self.model_units = ["K", "km/s", "km/s", "G", "G", "G"]


        # Denormalization bounds for Stokes parameters (added for plots)
        self.stokes_lower = [0.8,  -1e-2, -1e-2, -5e-2]
        self.stokes_upper = [1.2,   1e-2,  1e-2,  5e-2]
        self.stokes_units = ["I/I$_c$", "Q/I$_c$", "U/I$_c$", "V/I$_c$"]

        # logtau axis: linearly spaced from 1.5 to -7.5, 80 points
        self.logtau = np.linspace(1.5, -7.5, 80)

        # Depth cutoff: only use depths where logtau >= logtau_cutoff
        self.logtau_cutoff = -3  
        self.depth_cutoff = int(np.searchsorted(-self.logtau, -self.logtau_cutoff))
        # searchsorted on negated arrays because logtau is decreasing

        with h5py.File('../database/stokes_testing.h5', 'r') as f:
            self.wavelength = f['spec1']['wavelength'][:]

        
    def denormalize(self, models):
        lower = [2000., 0.0, -10.0, 0.0, -1000.0, -1000.0]
        upper = [25000., 3.0, 10.0, 1000.0, 1000.0, 1000.0]
        
        for i in range(models.shape[1]):
            models[:, i, :] = denormalize_output(models[:, i, :], lower[i], upper[i])

        return models
                
    def test(self):

        # Use four workers to load the data
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}

        # Testing dataset
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

        with torch.no_grad():

            for batch_idx, (stokes, models) in enumerate(t):

                # Move data to the computing device (GPU or CPU)
                models = models.to(self.device) # shape (batch size, 6, 80)
                stokes = stokes.to(self.device) # shape (batch size, 4, 112) 

                # L: einops.rearrange function is used to flatten the spatial/spectral 
                # dimensions of the input tensors into a single dimension, preparing them for 
                # input to the neural network encoders.
                # 'b c h -> b (c h)' keeps the batch dim. unchanged and flattens the last 
                # two dimensions c and h into a single dimension (c h)

                # This flattening is required because the ResNet encoders are defined with
                # specific input dimensions: they expect 1D feature vectors, not 2D spatial/spectral data.
                # The flattening converts each sample from a 2D profile (channels x spatial points) 
                # into a single long vector that the neural network can process.

                stokes_flat = rearrange(stokes, 'b c h -> b (c h)')
                models_flat = rearrange(models, 'b c h -> b (c h)')

                # Use encoder to get z_stokes and z_models
                z_s = self.encoder_stokes(stokes_flat)
                z_m = self.encoder_models(models_flat)

                z_s = F.normalize(z_s, dim=-1) # normalization along the last dimension ( the feature dimension), ensuring each embedding has unit length (norm=1)
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


        return z_stokes, z_models, models_all, stokes_all, decoded_models_all, decoded_stokes_all

    def plot_reconstruction(self, stokes, decoded_stokes, models, decoded_models, n_samples=3):
        """
        Plot the original (before latent space) vs decoded Stokes and model parameters
        for a few random samples to check how well reconstruction works.
        PENDING: IMPROVE THIS DESCRIPTION AS I DID WITH THE OTHERS.
        """

        n_total = stokes.shape[0] # number of available samples in the dataset
        # randomly choose which ones to visualize (without exceeding total)
        indices = random.sample(range(n_total), min(n_samples, n_total))

        stokes_labels = ["I", "Q", "U", "V"]
        model_labels = ["T", "vmic", "v", "Bx", "By", "Bz"]

        # create the directory to save the figures, with date in the name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = os.path.join(os.path.dirname(__file__), f"{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Saving validation plots to folder: {self.output_dir}")

        for idx in indices:
            fig, axes = pl.subplots(2,1, figsize=(10,8))
            #fig.suptitle(f'Sample {idx}', fontsize=14, fontweight='bold')
            fig.suptitle(f'Weights: w_clip=2, w_stokes=1, w_models=2')

            # plot the Stokes profiles
            ax = axes[0]
            for s in range(4): # Stoke parameter
                ax.plot(stokes[idx, s], label=f'{stokes_labels[s]}')
                ax.plot(decoded_stokes[idx, s], "--", label=f'{stokes_labels[s]} (decoded)')
            ax.set_title("Stokes profiles")
            ax.set_xlabel("Wavelength index")
            ax.set_ylabel("Normalized intensity")
            ax.legend(fontsize=8)

            # plot the model parameters
            ax = axes[1]
            for m in range(6):
                ax.plot(models[idx, m], label=f'{model_labels[m]}')
                ax.plot(decoded_models[idx, m], "--", label=f'{model_labels[m]} (decoded)')
            ax.set_title("Model parameters")
            ax.set_xlabel("Depth index")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8, ncol=3)

            pl.tight_layout(rect=[0, 0, 1, 0.96])
            #pl.show()
            #output_file = f"reconstruction_sample_{idx}.pdf"
            output_file = os.path.join(self.output_dir, f"reconstruction_{idx}.pdf")
            pl.savefig(output_file, dpi=150)
            print(f"Saved {output_file}")
            pl.close()
        

    def plot_tsne_joint(self, z_stokes, z_models, models, params=None, height_idx=40, use_pca=True, perplexity=30, depth_avg=False):
        """
        Joint t-SNE projection of z_stokes and z_models.
        Both latent spaces are embedded into the same 2D space.
        The t-SNE is computed once and reused across all panels; only the colormap changes.
        param: one of ["T", "vmic", "v", "Bx", "By", "Bz"]
        height_idx: atmospheric depth index (0-79)
        PENDING: IMPROVE THIS DESCRIPTION (a bit more) AS I DID WITH THE OTHERS.
        """

        # select a physical parameter
        param_dict  = {"T": 0, "vmic": 1, "v": 2, "Bx": 3, "By": 4, "Bz": 5}
        param_units = {"T": "K", "vmic": "km/s", "v": "km/s", "Bx": "G", "By": "G", "Bz": "G"}

        if params is None:
            params = ["T", "vmic", "v", "Bx", "By", "Bz"]

        # Step 1: compute the joint t-SNE embedding once
        # Both latent spaces are concatenated so that Stokes and model encoder
        # points are embedded into the same 2D

        print("Computing t-SNE projection...")

        z_all = np.concatenate([z_stokes, z_models], axis=0)  # shape: (2N, latent_dim)

        # pptional PCA pre-reduction (which is suggested in the documentation)
        # source: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        # NOTA: I disabled it (no longer default) because it didn't really make a difference

        if use_pca and z_all.shape[1] > 50:
            print("Applying PCA: reducing to 50 dims before t-SNE...")
            z_all = PCA(n_components=50).fit_transform(z_all)

        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                    learning_rate="auto", random_state=42)
        z_2d = tsne.fit_transform(z_all)  # (2N, 2)

        # split back for the plot
        N = len(z_stokes)
        z_stokes_2d = z_2d[:N]   # (N, 2)
        z_models_2d = z_2d[N:]   # (N, 2)

        print("t-SNE done. Plotting...")

        # Step 2: plot the space

        n_params = len(params)
        n_cols = 2
        n_rows = 3  # 3x2 fills a portrait page cleanly

        fig, axes = pl.subplots(n_rows, n_cols, figsize=(12, 16))
        axes = axes.flatten()

        for i, param in enumerate(params):
            ax = axes[i]
            p_index = param_dict[param]

            lo = self.model_lower[p_index]
            hi = self.model_upper[p_index]
            if depth_avg:
                values = denormalize_output(models[:, p_index, :].mean(axis=1), lo, hi)
                depth_label = "depth average"
            else:
                values = denormalize_output(models[:, p_index, height_idx], lo, hi)
                depth_label = f"log τ={self.logtau[height_idx]:.1f}"
            vmin, vmax = values.min(), values.max()

            # plot models encoder first (underneath), Stokes encoder on top
            ax.scatter(z_models_2d[:, 0], z_models_2d[:, 1],
                    c=values, cmap="viridis", s=15, marker="+",
                    alpha=0.5, linewidths=0.6,
                    vmin=vmin, vmax=vmax,
                    label="Models enc.")

            sc = ax.scatter(z_stokes_2d[:, 0], z_stokes_2d[:, 1],
                            c=values, cmap="viridis", s=15, marker="x",
                            alpha=0.5, linewidths=0.6,
                            vmin=vmin, vmax=vmax,
                            label="Stokes enc.")

            unit = param_units[param]
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            #cbar.set_label(f"{param} [{unit}] at log τ={self.logtau[height_idx]:.1f}", fontsize=9)
            cbar.set_label(f"{param} [{unit}] ({depth_label})", fontsize=9)

            ax.set_title(param, fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=8, markerscale=2.0, loc="upper right")

        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)

        #fig.suptitle(f"Joint latent space (t-SNE), perplexity={perplexity}", fontsize=13)
        pl.tight_layout(rect=[0, 0, 1, 0.97])

        if not hasattr(self, 'output_dir'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_dir = os.path.join(os.path.dirname(__file__), f"{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)

        #filename = os.path.join(self.output_dir, f"tsne_all_h{height_idx}.pdf")
        filename = os.path.join(self.output_dir, f"tsne_joint_all_{'davg' if depth_avg else f'h{height_idx}'}.pdf")
        pl.savefig(filename, dpi=150)
        print(f"Saved {filename}")
        pl.close()


    def fast_stokes_synthesis(self, models_all, stokes_all, n_ball=0, ball_sigma=0.02):
        """
        Fast Stokes synthesizer: Model -> encoder_models -> z -> decoder_stokes -> Stokes.

        Parameters:
        - models_all: shape (N, 6, 80) ; normalized model parameters
        - stokes_all: shape (N, 4, 112) ; normalized Stokes profiles (ground truth)
        - n_ball: number of perturbed z samples per profile for the ball 'experiment' (PENDING:think of better name). Set to 0 to skip it.
        - ball_sigma: standard dev. of the Gaussian perturbation applied to z for the ball.

        Returns a dictionary with:
        - 'synthesized_stokes': shape (N, 4, 112) ; predicted Stokes profiles
        - 'residuals': (N, 4, 112) ; residuals (pred - true)
        - 'rms_per_profile': (N, 4) ; RMS error per Stokes component per sample
        - 'rms_per_component': (4,) ; mean RMS across all samples per component
        - 'ball_profiles': (N, n_ball, 4, 112) or None
        - 'ball_mean': (N, 4, 112) or None — mean over ball samples
        - 'ball_std': (N, 4, 112) or None — std over ball samples
        """

        # in case use_decoders is set to False in conf.yaml
        if not self.decoders:
            raise RuntimeError("Remember to use_decoders=True in config for synthesis/inversion to work.")

        print("Running fast Stokes synthesis: model -> encoder_models -> z -> decoder_stokes ...")

        if n_ball > 0:
            print(f"Ball region enabled: n_ball={n_ball}, ball_sigma={ball_sigma}")

        # following the same steps as in test():
        # convert numpy arrays to tensors and build a DataLoader, so we process in batches
        models_tensor = torch.tensor(models_all, dtype=torch.float32)
        stokes_tensor = torch.tensor(stokes_all, dtype=torch.float32)

        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(models_tensor, stokes_tensor), batch_size=self.batch_size, shuffle=False)

        synthesized_list = []
        ball_list = []  # will stay empty if n_ball=0

        with torch.no_grad():
            for models_batch, stokes_batch in tqdm(loader):

                models_batch = models_batch.to(self.device)
                B = models_batch.shape[0]

                # flatten (I added explanation above): (B, 6, 80) -> (B, 480) 
                models_flat = rearrange(models_batch, 'b c h -> b (c h)')

                # encode physical models into the shared latent space
                z = self.encoder_models(models_flat) # (B, latent_dim)
                z = F.normalize(z, dim=-1)  # unit norm

                # then decode as Stokes
                synth_flat = self.decoder_stokes(z)

                # reshape back to (B, 4, 112)
                synth = rearrange(synth_flat, 'b (c h) -> b c h', c=4)
                synthesized_list.append(synth.cpu().numpy())

                # Ball "experiment"? Otherwise known as "lo de la bola"
                # For each sample in the batch, we draw n_ball small Gaussian perturbations
                # around its latent point z and decode each one. This  will show us how
                # sensitive the output Stokes profiles are to small movements in latent space.
                if n_ball > 0:
                    # expand z to (B, n_ball, latent_dim) and add Gaussian noise
                    z_expanded = z.unsqueeze(1).expand(B, n_ball, -1) # insert new dim. and copy each sample n_ball times
                    # https://docs.pytorch.org/docs/main/generated/torch.unsqueeze.html 
                    # "Returns a new tensor with a dimension of size one inserted at the specified position.
                    noise = torch.randn_like(z_expanded)*ball_sigma # create random noise with the same shape

                    # re-normalise after perturbation to keep points on the unit sphere 
                    # OJO: Skipping this would feed the decoder inputs it has never seen, because the decoder was trained on normalized vectors.
                    z_perturbed = F.normalize(z_expanded + noise, dim=-1)

                    # flatten to (B*n_ball, latent_dim) from (B, n_ball, latent_dim) to be able to decode in a sigle pass
                    z_perturbed_flat = rearrange(z_perturbed, 'b n d -> (b n) d')

                    # decode all perturbed points 
                    ball_flat = self.decoder_stokes(z_perturbed_flat)  # (B*n_ball, 4*112)

                    # reshape to (B, n_ball, 4, 112)
                    ball_profiles = rearrange(ball_flat, '(b n) (c h) -> b n c h', b=B, c=4)
                    ball_list.append(ball_profiles.cpu().numpy())


        # concatenate across all batches
        synthesized_stokes = np.concatenate(synthesized_list, axis=0)  # (N, 4, 112)

        # residuals: difference between synthesized and ground-truth Stokes
        residuals = synthesized_stokes - stokes_all  # (N, 4, 112)

        # RMS per sample per Stokes component: sqrt(mean over wavelength axis)
        rms_per_profile = np.sqrt(np.mean(residuals**2, axis=2))  # (N, 4)

        # mean RMS over the entire test set for each Stokes component
        rms_per_component = np.mean(rms_per_profile, axis=0)  # (4,)

        stokes_labels = ["I", "Q", "U", "V"]
        print("\n--- Fast Synthesis RMS---")
        for i, label in enumerate(stokes_labels): print(f"  Stokes {label}: {rms_per_component[i]:.5f}")

        # reunite all ball results
        if n_ball > 0:
            ball_profiles_all = np.concatenate(ball_list, axis=0) # (N, n_ball, 4, 112)
            ball_mean = ball_profiles_all.mean(axis=1) # (N, 4, 112)
            ball_std = ball_profiles_all.std(axis=1)  # (N, 4, 112)
        else:
            ball_profiles_all = None
            ball_mean = None
            ball_std  = None

        return {
            'synthesized_stokes': synthesized_stokes,
            'residuals': residuals,
            'rms_per_profile':rms_per_profile,
            'rms_per_component': rms_per_component,
            'ball_profiles': ball_profiles_all,
            'ball_mean': ball_mean,
            'ball_std': ball_std,
        }


    def plot_fast_synthesis_results(self, stokes_all, synthesis_results, n_samples=3, indices=None):

        synthesized_stokes = synthesis_results['synthesized_stokes']
        residuals = synthesis_results['residuals']
        rms_per_profile = synthesis_results['rms_per_profile']
        rms_per_component  = synthesis_results['rms_per_component']
        ball_profiles = synthesis_results['ball_profiles']
        ball_mean = synthesis_results['ball_mean']
        ball_std = synthesis_results['ball_std']

        has_ball = ball_profiles is not None
        n_ball = ball_profiles.shape[1] if has_ball else 0

        stokes_labels = [r"I", r"Q/I$_c$", r"U/I$_c$", r"V/I$_c$"]
        N = stokes_all.shape[0]

        if not hasattr(self, 'output_dir'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_dir = os.path.join(os.path.dirname(__file__), f"{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)

        if indices is None:
            indices = random.sample(range(N), min(n_samples, N))

        # precompute RMS in physical units for every sample and component
        # shape: (N, 4)
        rms_phys_per_profile = np.zeros_like(rms_per_profile)
        for s in range(4):
            lo, hi = self.stokes_lower[s], self.stokes_upper[s]
            pred_phys = denormalize_output(synthesized_stokes[:, s, :], lo, hi)
            gt_phys = denormalize_output(stokes_all[:, s, :],         lo, hi)
            rms_phys_per_profile[:, s] = np.sqrt(np.mean((pred_phys - gt_phys)**2, axis=1))

        # mean RMS in physical units across all samples
        rms_phys_per_component = rms_phys_per_profile.mean(axis=0)  # (4,)

        # ------------------------------------------------------------------
        # Figure 1: Profile comparisons
        # ------------------------------------------------------------------
        for idx in indices:
            fig, axes = pl.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

            for s, label in enumerate(stokes_labels):
                ax = axes[s]
                lo, hi = self.stokes_lower[s], self.stokes_upper[s]

                gt   = denormalize_output(stokes_all[idx, s], lo, hi)
                pred = denormalize_output(synthesized_stokes[idx, s], lo, hi)

                if has_ball:
                    for b in range(n_ball):
                        bp = denormalize_output(ball_profiles[idx, b, s], lo, hi)
                        ax.plot(self.wavelength, bp, color='lightblue', alpha=0.15, linewidth=0.4)

                    bm = denormalize_output(ball_mean[idx, s], lo, hi)
                    bs = ball_std[idx, s] * 0.5 * (hi - lo)
                    ax.fill_between(self.wavelength, bm - bs, bm + bs, color='steelblue', alpha=0.35, label='Ball ±1σ')
                    ax.plot(self.wavelength, bm, color='steelblue', linewidth=1.5, linestyle='--', label='Ball mean')

                ax.plot(self.wavelength, pred, color='red',   linewidth=1.5, linestyle='--', label='Synthesized')
                ax.plot(self.wavelength, gt, color='black', linewidth=1.5, label='Ground truth')

                rms_val = rms_phys_per_profile[idx, s]
                ax.set_title(f"{label}   (RMS = {rms_val:.4f})", fontsize=13)
                ax.set_xlabel("Wavelength [Å]", fontsize=12)
                ax.set_ylabel(label, fontsize=12)
                ax.tick_params(labelsize=11)
                ax.legend(fontsize=10)

            pl.tight_layout(rect=[0, 0, 1, 0.95])
            out = os.path.join(self.output_dir, f"synthesis_profiles_{idx}.pdf")
            pl.savefig(out, dpi=150)
            print(f"Saved {out}")
            pl.close()

        # ------------------------------------------------------------------
        # Figure 2: Residual distributions 
        # ------------------------------------------------------------------
        fig, axes = pl.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for s, label in enumerate(stokes_labels):
            ax = axes[s]
            lo, hi = self.stokes_lower[s], self.stokes_upper[s]

            pred_phys = denormalize_output(synthesized_stokes[:, s, :], lo, hi)
            gt_phys = denormalize_output(stokes_all[:, s, :], lo, hi)
            res_flat = (pred_phys - gt_phys).flatten()

            ax.hist(res_flat, bins=80, color='rebeccapurple', edgecolor='none', density=True)
            ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
            ax.set_title(label, fontsize=13)
            ax.set_xlabel(f"Residual [{label}]", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.tick_params(labelsize=11)
            ax.text(0.97, 0.95, f"μ = {res_flat.mean():.4f}\nσ = {res_flat.std():.4f}", transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        pl.tight_layout()
        out = os.path.join(self.output_dir, "synthesis_residuals.pdf")
        pl.savefig(out, dpi=150)
        print(f"Saved {out}")
        pl.close()

        # ------------------------------------------------------------------
        # Figure 3: RMS bar chart 
        # ------------------------------------------------------------------
        fig, ax = pl.subplots(figsize=(7, 5))
        bars = ax.bar(stokes_labels, rms_phys_per_component, color=['steelblue', 'coral', 'mediumseagreen', 'orchid'])

        for bar, val in zip(bars, rms_phys_per_component):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.01, f"{val:.4f}", ha='center', va='bottom', fontsize=11)

        ax.set_xlabel("Stokes parameter", fontsize=13)
        ax.set_ylabel("Mean RMS (physical units)", fontsize=13)
        ax.tick_params(labelsize=12)
        pl.tight_layout()
        out = os.path.join(self.output_dir, "synthesis_rms_summary.pdf")
        pl.savefig(out, dpi=150)
        print(f"Saved {out}")
        pl.close()



    def fast_stokes_inversion(self, stokes_all, models_all, n_ball=100, ball_sigma=0.02):
        """
        Fast Stokes inverter: Stokes -> encoder_stokes -> z -> decoder_models -> model.

        Parameters:
        - stokes_all: shape (N, 4, 112) — normalized Stokes profiles (input)
        - models_all:  shape (N, 6, 80)  — normalized model parameters (ground truth)
        - n_ball: number of perturbed z samples per profile. Set to 0 to skip.
        - ball_sigma: standard dev. of the Gaussian perturbation applied to z.

        Returns a dictionary with:
        - 'inverted_models': (N, 6, 80) ;  predicted physical parameters
        - 'residuals': (N, 6, 80) ; residuals (pred - true)
        - 'rms_per_profile': (N, 6) ; RMS per model component per sample
        - 'rms_per_component': (6,) ;  mean RMS across all samples
        - 'ball_profiles': (N, n_ball, 6, 80) or None
        - 'ball_mean': (N, 6, 80) or None ; mean over ball samples
        - 'ball_std': (N, 6, 80) or None ; std over ball samples
        - 'ball_sigma': float ; sigma used (stored for plot titles)
        """

        if not self.decoders:
            raise RuntimeError("Remember to use_decoders=True in config for synthesis/inversion to work.")

        print("Running fast Stokes inversion: Stokes -> encoder_stokes -> z -> decoder_models ...")
        if n_ball > 0:
            print(f"Ball experiment enabled: n_ball={n_ball}, ball_sigma={ball_sigma}")

        # Convert numpy arrays to tensors and build a DataLoader so we can process in batches
        stokes_tensor = torch.tensor(stokes_all, dtype=torch.float32)
        models_tensor = torch.tensor(models_all, dtype=torch.float32)

        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(stokes_tensor, models_tensor), batch_size=self.batch_size,shuffle=False)

        inverted_list = []
        ball_list = []

        with torch.no_grad():
            for stokes_batch, _ in tqdm(loader, desc="Inverting Stokes"):

                stokes_batch = stokes_batch.to(self.device)
                B = stokes_batch.shape[0]

                # flatten: (B, 4, 112) -> (B, 448)
                stokes_flat = rearrange(stokes_batch, 'b c h -> b (c h)')

                # encode Stokes profiles into the shared latent space
                z = self.encoder_stokes(stokes_flat)
                z = F.normalize(z, dim=-1)  # unit-norm, consistent with training

                # decoded as physical model and reshape back to original shape
                inverted_flat = self.decoder_models(z)
                inverted = rearrange(inverted_flat, 'b (c h) -> b c h', c=6)
                inverted_list.append(inverted.cpu().numpy())

                # Ball "experiment"? Otherwise known as "lo de la bola"
                # For each sample in the batch, we draw n_ball small gaussian perturbations
                # around its latent point z and decode each one. This  will show us how
                # sensitive the output Stokes profiles are to small movements in latent space.
                if n_ball > 0:
                    # expand z to (B, n_ball, latent_dim) and add Gaussian noise
                    z_expanded = z.unsqueeze(1).expand(B, n_ball, -1)
                    noise = torch.randn_like(z_expanded) * ball_sigma

                    # re-normalise after perturbation to keep points on the unit sphere
                    z_perturbed = F.normalize(z_expanded + noise, dim=-1)

                    # flatten to (B*n_ball, latent_dim) for a single forward pass
                    z_perturbed_flat = rearrange(z_perturbed, 'b n d -> (b n) d')

                    # decode all perturbed points with decoder_models
                    ball_flat = self.decoder_models(z_perturbed_flat)  # (B*n_ball, 6*80)

                    # reshape to (B, n_ball, 6, 80)
                    ball_profiles = rearrange(ball_flat, '(b n) (c h) -> b n c h', b=B, c=6)
                    ball_list.append(ball_profiles.cpu().numpy())

        inverted_models = np.concatenate(inverted_list, axis=0)  # (N, 6, 80)

        # residuals and RMS
        residuals = inverted_models - models_all # (N, 6, 80)
        dc=self.depth_cutoff
        rms_per_profile = np.sqrt(np.mean(residuals[:, :, :dc]**2, axis=2))  # (N, 6)
        rms_per_component = np.mean(rms_per_profile, axis=0)  # (6,)

        model_labels = ["T", "vmic", "v", "Bx", "By", "Bz"]
        print("\n--- Fast Inversion RMS (normalized units) ---")
        for i, label in enumerate(model_labels): print(f"  {label}: {rms_per_component[i]:.5f}")

        # assemble all the ball results
        if n_ball > 0:
            ball_profiles_all = np.concatenate(ball_list, axis=0)   # (N, n_ball, 6, 80)
            ball_mean = ball_profiles_all.mean(axis=1) # (N, 6, 80)
            ball_std = ball_profiles_all.std(axis=1)  # (N, 6, 80)
        else:
            ball_profiles_all = None
            ball_mean = None
            ball_std = None

        return {
            'inverted_models': inverted_models,
            'residuals': residuals,
            'rms_per_profile': rms_per_profile,
            'rms_per_component':rms_per_component,
            'ball_profiles': ball_profiles_all,
            'ball_mean': ball_mean,
            'ball_std': ball_std,
            'ball_sigma': ball_sigma,
        }


    def plot_fast_inversion_results(self, models_all, inversion_results, n_samples=3, indices=None):

        inverted_models = inversion_results['inverted_models']
        residuals = inversion_results['residuals']
        rms_per_profile = inversion_results['rms_per_profile']
        rms_per_component = inversion_results['rms_per_component']
        ball_profiles = inversion_results['ball_profiles']
        ball_mean = inversion_results['ball_mean']
        ball_std = inversion_results['ball_std']
        ball_sigma = inversion_results['ball_sigma']

        has_ball = ball_profiles is not None
        n_ball = ball_profiles.shape[1] if has_ball else 0

        model_labels = ["T", "v$_\mathrm{mic}$", "v", "B$_x$", "B$_y$", "B$_z$"]
        colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid', 'goldenrod', 'tomato']
        N = models_all.shape[0]

        depth_mask = self.logtau >= self.logtau_cutoff
        logtau_plot= self.logtau[depth_mask]

        if not hasattr(self, 'output_dir'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_dir = os.path.join(os.path.dirname(__file__), f"{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)

        if indices is None:
            indices = random.sample(range(N), min(n_samples, N))

        # precompute RMS in physical units for every sample and component
        # RMS is computed only over the reliable depth range (depth_mask)
        rms_phys_per_profile = np.zeros_like(rms_per_profile)  # (N, 6)
        for m in range(6):
            lo, hi = self.model_lower[m], self.model_upper[m]
            pred_phys = denormalize_output(inverted_models[:, m, :][:, depth_mask], lo, hi)
            gt_phys = denormalize_output(models_all[:, m, :][:, depth_mask],      lo, hi)
            rms_phys_per_profile[:, m] = np.sqrt(np.mean((pred_phys - gt_phys)**2, axis=1))

        rms_phys_per_component = rms_phys_per_profile.mean(axis=0)  # (6,)

        # ------------------------------------------------------------------
        # Figure 1: Profile comparisons
        # ------------------------------------------------------------------
        for idx in indices:
            fig, axes = pl.subplots(2, 3, figsize=(15, 9))
            axes = axes.flatten()

            for m, label in enumerate(model_labels):
                ax = axes[m]
                lo, hi = self.model_lower[m], self.model_upper[m]

                gt = denormalize_output(models_all[idx, m][depth_mask], lo, hi)
                pred = denormalize_output(inverted_models[idx, m][depth_mask], lo, hi)

                if has_ball:
                    for b in range(n_ball):
                        bp = denormalize_output(ball_profiles[idx, b, m][depth_mask], lo, hi)
                        ax.plot(logtau_plot, bp, color='lightblue', alpha=0.15, linewidth=0.4)

                    bm = denormalize_output(ball_mean[idx, m][depth_mask], lo, hi)
                    bs = ball_std[idx, m][depth_mask] * 0.5 * (hi - lo)
                    ax.fill_between(logtau_plot, bm - bs, bm + bs, color='steelblue', alpha=0.35, label='Ball ±1σ')
                    ax.plot(logtau_plot, bm, color='steelblue', linewidth=1.5, linestyle='--', label='Ball mean')

                ax.plot(logtau_plot, pred, color='red',   linewidth=1.5, linestyle='--', label='Inverted')
                ax.plot(logtau_plot, gt, color='black', linewidth=1.5, label='Ground truth')

                rms_val = rms_phys_per_profile[idx, m]
                ax.set_title(f"{label}   (RMS = {rms_val:.2f} {self.model_units[m]})", fontsize=13)
                ax.set_xlabel("log τ", fontsize=12)
                ax.set_ylabel(f"{label} [{self.model_units[m]}]", fontsize=12)
                ax.tick_params(labelsize=11)
                ax.invert_xaxis()
                ax.legend(fontsize=10)

                if label == "T":
                    ax.set_ylim(2000, 10000)

            pl.tight_layout(rect=[0, 0, 1, 0.95])
            out = os.path.join(self.output_dir, f"inversion_profiles_{idx}.pdf")
            pl.savefig(out, dpi=150)
            print(f"Saved {out}")
            pl.close()

        # ------------------------------------------------------------------
        # Figure 2: Residual distributions 
        # ------------------------------------------------------------------
        fig, axes = pl.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for m, label in enumerate(model_labels):
            ax = axes[m]
            lo, hi = self.model_lower[m], self.model_upper[m]

            pred_phys = denormalize_output(inverted_models[:, m, :][:, depth_mask], lo, hi)
            gt_phys = denormalize_output(models_all[:, m, :][:, depth_mask], lo, hi)
            res_flat = (pred_phys - gt_phys).flatten()

            ax.hist(res_flat, bins=80, color='rebeccapurple', edgecolor='none', density=True)
            ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
            ax.set_title(label, fontsize=13)
            ax.set_xlabel(f"Residual [{self.model_units[m]}]", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.tick_params(labelsize=11)
            ax.text(0.97, 0.95, f"μ = {res_flat.mean():.2f}\nσ = {res_flat.std():.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        pl.tight_layout()
        out = os.path.join(self.output_dir, "inversion_residuals.pdf")
        pl.savefig(out, dpi=150)
        print(f"Saved {out}")
        pl.close()

        # ------------------------------------------------------------------
        # Figure 3: RMS bar chart 
        # ------------------------------------------------------------------
        unit_labels = [f"{lb}\n[{u}]" for lb, u in zip(["T", "v$_\mathrm{mic}$", "v", "B$_x$", "B$_y$", "B$_z$"], self.model_units)]

        fig, ax = pl.subplots(figsize=(9, 5))
        bars = ax.bar(unit_labels, rms_phys_per_component, color=colors)

        for bar, val in zip(bars, rms_phys_per_component): ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.01, f"{val:.2f}", ha='center', va='bottom', fontsize=11)

        ax.set_xlabel("Model parameter", fontsize=13)
        ax.set_ylabel("Mean RMS (physical units)", fontsize=13)
        ax.tick_params(labelsize=12)
        pl.tight_layout()
        out = os.path.join(self.output_dir, "inversion_rms_summary.pdf")
        pl.savefig(out, dpi=150)
        print(f"Saved {out}")
        pl.close()


if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*.pth')   
    files.sort()
    #checkpoint = files[-1]
    checkpoint = '../train/weights/2026-03-26-20_21_50_clip.pth' # (w_clip=2, w_stokes=1, w_models=2), noise=1e-3
    #checkpoint = '../train/weights/2026-05-05-11_54_48_clip.pth' # (w_clip=0, w_stokes=1, w_models=1), noise=1e-3
    

    deepnet = Testing(checkpoint, gpu=0, batch_size=1024)
    z_stokes, z_models, models, stokes, decoded_models, decoded_stokes = deepnet.test()

    # Autoencoder
    #deepnet.plot_reconstruction(stokes, decoded_stokes, models, decoded_models, n_samples=3)

    # using method defined in dataset.py (RMS as metric for S/N)
    #fixed_indices = [5116, 4844, 3553, 5579, 2586, 5262, 3008, 3858, 810, 1610] # 10 samples with highest S/N, using RMS metric
    #fixed_indices = [590, 962, 2605, 2634, 4906, 892, 3693, 4421, 3267, 2333] # 10 samples with mid S/N, using RMS metric
    #fixed_indices = [521, 3633, 3636, 4923, 881, 427, 1332, 4492, 540, 3296] # 10 samples with low S/N, using RMS metric

    # using noise_study method 
    #fixed_indices = [2123, 3802, 1410, 2087, 5733, 1318, 2320, 3719, 1436, 1020]#[4362, 2728, 3774] # [4362, 2728, 3774] are representative samples (closest to the median S/N, when including I in its computation)
    #fixed_indices = [4684, 1978, 4395, 2512, 2112, 607, 4894, 5423, 5548, 2556] # representative, (closest to median S/N=265.9), including I in computation
    #fixed_indices = [445, 521, 2502, 427, 1225] #[4492, 521, 540] # these should be some of the lowest S/N samples, let's see if it's true. It seems true lol
    #fixed_indices = [367, 5413, 898, 3889, 1104] #[719, 5413, 1157] # highest S/N samples (without seed, excliding I from the median calculation)
    

    # NEW VERSION: I fixed the seed in noise_study.py
    # high S/N
    #fixed_indices = [5413, 3426, 4347, 5637, 2719, 4993, 4976, 5996, 3959, 400]
    # median S/N
    #fixed_indices = [2298, 2622, 2600, 2734, 3420, 2232, 3252, 1539, 4333, 1192]
    # low S/N
    #fixed_indices = [521, 1053, 4895, 3636, 427, 3296, 5170, 987, 3633, 2380]

    # Chosen for analysis (for now)
    fixed_indices = [3426, 987, 3252]
    
    # Fast Stokes synthesis
    synthesis_results = deepnet.fast_stokes_synthesis(models, stokes, n_ball=100, ball_sigma=0.02)    
    deepnet.plot_fast_synthesis_results(stokes, synthesis_results, n_samples=3, indices=fixed_indices)

    # Fast Stokes inversion
    inversion_results = deepnet.fast_stokes_inversion(stokes, models, n_ball=100, ball_sigma=0.02)
    deepnet.plot_fast_inversion_results(models, inversion_results, n_samples=3, indices=fixed_indices)

    # t-SNE representation of latent space
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, height_idx=20, use_pca=False, depth_avg=False)
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, height_idx=40, use_pca=False, depth_avg=False)
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, height_idx=60, use_pca=False, depth_avg=False)

    #groups = {
    #'high':   [5413, 3426, 4347, 5637, 2719, 4993, 4976, 5996, 3959, 400],
    #'median': [2298, 2622, 2600, 2734, 3420, 2232, 3252, 1539, 4333, 1192],
    #'low':    [521, 1053, 4895, 3636, 427, 3296, 5170, 987, 3633, 2380],
    #}



