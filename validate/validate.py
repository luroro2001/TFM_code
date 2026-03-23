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
#import matplotlib
#matplotlib.use('Agg') # test, did not fix the issue
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
#import time
#import pathlib
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

            #L: also put decoders in evaluation mode for plot_reconstruction
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

        with torch.no_grad():

            for batch_idx, (stokes, models) in enumerate(t):
                models = models.to(self.device) # shape (batch size, 6, 80)
                stokes = stokes.to(self.device) # shape (batch size, 4, 112) 

                # L: use the einops.rearrange function to flatten the spatial/spectral 
                # dimensions of the input tensors into a single dimension, preparing them for 
                # input to the neural network encoders.
                # 'b c h -> b (c h)' keeps the batch dim. unchanged and flattens the last 
                # two dimensions c and h into a single dimension (c h)

                # This flattening is required because the ResNet encoders are defined with
                # specific input dimensions: they expect 1D feature vectors, not 2D spatial/spectral data.
                # The flattening converts each sample from a 2D profile (channels × spatial points) 
                # into a single long vector that the neural network can process.

                stokes_flat = rearrange(stokes, 'b c h -> b (c h)')
                models_flat = rearrange(models, 'b c h -> b (c h)')

                z_s = self.encoder_stokes(stokes_flat)
                z_m = self.encoder_models(models_flat)

                z_s = F.normalize(z_s, dim=-1) # L2 normalization along the last dimension ( the feature dimension), ensuring each embedding has unit length (norm=1)
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

        # THIS CAUSES THE MODEL PARAMS TO BE IN PHYSICAL UNITS IN THE PLOT
        #models_all = self.denormalize(models_all)
        #decoded_models_all = self.denormalize(decoded_models_all) if self.decoders else None

        return z_stokes, z_models, models_all, stokes_all, decoded_models_all, decoded_stokes_all

    def plot_reconstruction(self, stokes, decoded_stokes, models, decoded_models, n_samples=3):
        """
        Plot the original (before latent space) vs decoded Stokes and model parameters
        for a few random samples to check how well reconstruction works.
        """

        n_total = stokes.shape[0] # number of available samples in dataset
        # randomly choose which ones to visualize (without exceeding total)
        indices = random.sample(range(n_total), min(n_samples, n_total))

        stokes_labels = ["I", "Q", "U", "V"]
        model_labels = ["T", "vmic", "v", "Bx", "By", "Bz"]

        # create the directory to save the figures, with date in the name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = os.path.join(os.path.dirname(__file__), f"weight_trial_{timestamp}")
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
        

    def plot_tsne_joint(self, z_stokes, z_models, models, param="Bz", height_idx=40, use_pca=True, perplexity=30):
        """
        Joint t-SNE projection of z_stokes and z_models.
        Both latent spaces are embedded into the same 2D space
        param: one of ["T", "vmic", "v", "Bx", "By", "Bz"]
        height_idx: atmospheric depth index (0–79)
        """

        print("Running joint t-SNE projection...")

        # select physical parameter
        param_dict = {"T": 0, "vmic": 1, "v": 2, "Bx": 3, "By": 4, "Bz": 5}

        p_index = param_dict[param]
        # (extract the value of the physical parameter at a specific height)
        values = models[:, p_index, height_idx] # used to colour by chosen physical param. at chosen depth in plot

        # concatenate the latent spaces
        z_all = np.concatenate([z_stokes, z_models], axis=0) #shape is z_all : (2N, latent_dim)
        #print(z_all.shape) #(12186, 64)

        # pptional PCA pre-reduction
        # source: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        if use_pca and z_all.shape[1] > 50:
            print("Applying PCA: reducing to 50 dims before t-SNE")
            z_all = PCA(n_components=50).fit_transform(z_all)

        # compute joint t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)

        z_2d = tsne.fit_transform(z_all)

        # split back for plot
        N = len(z_stokes)
        z_stokes_2d = z_2d[:N]
        z_models_2d = z_2d[N:]

        # Plot
        pl.figure(figsize=(8, 7))

        # Stokes encoder
        sc1 = pl.scatter(z_stokes_2d[:, 0], z_stokes_2d[:, 1], c=values, cmap="viridis", s=10, marker="x", alpha=0.8, linewidths=0.5, label="Stokes encoder")

        # Models encoder
        sc2 = pl.scatter(z_models_2d[:, 0], z_models_2d[:, 1], c=values, cmap="viridis", s=10, marker="+", alpha=0.8, linewidths=0.5, label="Models encoder")

        pl.colorbar(sc1, label=f"{param} at depth {height_idx}")
        #pl.title(f"Joint latent space (t-SNE)\nColored by {param}")
        pl.title(f"Latent space (t-SNE)")
        pl.legend()
        pl.tight_layout()

        filename = os.path.join(self.output_dir, f"tsne_joint_{param}_h{height_idx}.pdf")

        pl.savefig(filename, dpi=150)
        print(f"Saved {filename}")
        pl.close()


    def fast_stokes_synthesis(self, models_all, stokes_all):
        """
        Fast Stokes synthesizer: Model -> z -> decoder_stokes -> Stokes.

        For each physical model in the test set, encodes it into the latent space
        using encoder_models, then decodes back to Stokes profiles using decoder_stokes.
        This cross-modal decode is only meaningful if the contrastive training has 
        successfully aligned both encoders in the shared latent space.

        Args:
            models_all : np.ndarray, shape (N, 6, 80)  — normalized model parameters
            stokes_all : np.ndarray, shape (N, 4, 112) — normalized Stokes profiles (ground truth)

        Returns:
            dict with:
                'synthesized_stokes' : np.ndarray (N, 4, 112) — predicted Stokes profiles
                'residuals'          : np.ndarray (N, 4, 112) — pointwise residuals (pred - true)
                'rms_per_profile'    : np.ndarray (N, 4)      — RMS error per Stokes component per sample
                'rms_per_component'  : np.ndarray (4,)        — mean RMS across all samples per component
        """

        if not self.decoders:
            raise RuntimeError("fast_stokes_synthesis() requires decoders (use_decoders=True in config).")

        print("Running fast Stokes synthesis: model -> z -> decoder_stokes ...")

        # Convert numpy arrays to tensors and build a DataLoader
        # so we process in batches (consistent with how test() works)
        models_tensor = torch.tensor(models_all, dtype=torch.float32)
        stokes_tensor = torch.tensor(stokes_all, dtype=torch.float32)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(models_tensor, stokes_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

        synthesized_list = []

        with torch.no_grad():
            for models_batch, stokes_batch in tqdm(loader, desc="Synthesizing Stokes"):

                models_batch = models_batch.to(self.device)

                # Flatten: (B, 6, 80) -> (B, 480) — mirrors the rearrange in train/test
                models_flat = rearrange(models_batch, 'b c h -> b (c h)')

                # Encode physical models into the shared latent space
                z = self.encoder_models(models_flat)
                z = F.normalize(z, dim=-1)  # unit-norm, consistent with training

                # Cross-modal decode: encoded from models, decoded as Stokes
                synth_flat = self.decoder_stokes(z)

                # Reshape back to (B, 4, 112)
                synth = rearrange(synth_flat, 'b (c h) -> b c h', c=4)
                synthesized_list.append(synth.cpu().numpy())

        synthesized_stokes = np.concatenate(synthesized_list, axis=0)  # (N, 4, 112)

        # Residuals: pointwise difference between synthesized and ground-truth Stokes
        residuals = synthesized_stokes - stokes_all  # (N, 4, 112)

        # RMS per sample per Stokes component: sqrt(mean over wavelength axis)
        rms_per_profile = np.sqrt(np.mean(residuals**2, axis=2))  # (N, 4)

        # Mean RMS over the entire test set for each Stokes component
        rms_per_component = np.mean(rms_per_profile, axis=0)  # (4,)

        stokes_labels = ["I", "Q", "U", "V"]
        print("\n--- Fast Synthesis RMS (normalized units) ---")
        for i, label in enumerate(stokes_labels):
            print(f"  Stokes {label}: {rms_per_component[i]:.5f}")

        return {
            'synthesized_stokes': synthesized_stokes,
            'residuals': residuals,
            'rms_per_profile': rms_per_profile,
            'rms_per_component': rms_per_component,
        }


    def plot_fast_synthesis_results(self, stokes_all, synthesis_results, n_samples=3):
        """
        Produces analysis plots for the fast Stokes synthesis.

        Generates three figure types:
        1) Profile comparisons: ground-truth vs synthesized Stokes for n_samples random profiles.
        2) Residual distributions: histogram of residuals for each Stokes component.
        3) RMS summary bar chart: mean RMS per Stokes component across the test set.
        4) RMS box plots: distribution of per-profile RMS values per Stokes component.

        Args:
            stokes_all        : np.ndarray (N, 4, 112) — ground-truth normalized Stokes profiles
            synthesis_results : dict returned by fast_stokes_synthesis()
            n_samples         : int — number of random profiles to plot in the comparison figure
        """

        synthesized_stokes = synthesis_results['synthesized_stokes']
        residuals          = synthesis_results['residuals']
        rms_per_profile    = synthesis_results['rms_per_profile']
        rms_per_component  = synthesis_results['rms_per_component']

        stokes_labels = ["I", "Q", "U", "V"]
        N = stokes_all.shape[0]

        # Use or create the output directory (consistent with plot_reconstruction)
        if not hasattr(self, 'output_dir'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_dir = os.path.join(os.path.dirname(__file__), f"weight_trial_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Saving synthesis plots to folder: {self.output_dir}")

        # ------------------------------------------------------------------
        # Figure 1: Profile comparisons (ground truth vs synthesized)
        # One figure per random sample, 4 subplots (one per Stokes component)
        # ------------------------------------------------------------------
        indices = random.sample(range(N), min(n_samples, N))

        for idx in indices:
            fig, axes = pl.subplots(2, 2, figsize=(12, 8))
            #fig.suptitle(f"Fast Stokes Synthesis — Sample {idx} (model → z → decoder_stokes)")
            axes = axes.flatten()

            for s, label in enumerate(stokes_labels):
                ax = axes[s]
                ax.plot(stokes_all[idx, s],         color='black', linewidth=1.5, label='Ground truth')
                ax.plot(synthesized_stokes[idx, s], color='red',   linewidth=1.5, linestyle='--', label='Synthesized')
                ax.set_title(f"Stokes {label}  (RMS={rms_per_profile[idx, s]:.4f})")
                ax.set_xlabel("Wavelength index")
                ax.set_ylabel("Normalized value")
                ax.legend(fontsize=8)

            pl.tight_layout(rect=[0, 0, 1, 0.95])
            out = os.path.join(self.output_dir, f"synthesis_profiles_{idx}.pdf")
            pl.savefig(out, dpi=150)
            print(f"Saved {out}")
            pl.close()

        # ------------------------------------------------------------------
        # Figure 2: Residual distributions (one subplot per Stokes component)
        # Histograms show whether errors are centred at zero and reveal outliers
        # ------------------------------------------------------------------
        fig, axes = pl.subplots(1, 4, figsize=(16, 4))
        #fig.suptitle("Residual distributions (synthesized − ground truth)")

        for s, label in enumerate(stokes_labels):
            ax = axes[s]
            res_flat = residuals[:, s, :].flatten()
            ax.hist(res_flat, bins=80, color='steelblue', edgecolor='none', density=True)
            ax.axvline(0, color='black', linestyle='--', linewidth=1.0)
            ax.set_title(f"Stokes {label}")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Density")

            # Annotate with mean and std for quick inspection
            ax.text(0.97, 0.95,
                    f"μ={res_flat.mean():.4f}\nσ={res_flat.std():.4f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        pl.tight_layout()
        out = os.path.join(self.output_dir, "synthesis_residuals.pdf")
        pl.savefig(out, dpi=150)
        print(f"Saved {out}")
        pl.close()

        # ------------------------------------------------------------------
        # Figure 3: RMS summary bar chart
        # One bar per Stokes component — easy headline metric to report
        # ------------------------------------------------------------------
        fig, ax = pl.subplots(figsize=(6, 4))
        bars = ax.bar(stokes_labels, rms_per_component,
                    color=['steelblue', 'coral', 'mediumseagreen', 'orchid'])

        for bar, val in zip(bars, rms_per_component):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.0002,
                    f"{val:.5f}", ha='center', va='bottom', fontsize=9)

        #ax.set_title("Mean RMS per Stokes component\n(Fast synthesis: model → z → decoder_stokes)")
        ax.set_xlabel("Stokes parameter")
        ax.set_ylabel("Mean RMS (normalized units)")
        pl.tight_layout()
        out = os.path.join(self.output_dir, "synthesis_rms_summary.pdf")
        pl.savefig(out, dpi=150)
        print(f"Saved {out}")
        pl.close()

        # ------------------------------------------------------------------
        # Figure 4: RMS box plots per Stokes component
        # Shows spread of per-profile RMS — useful for spotting whether failures
        # are isolated outliers or spread systematically across the test set
        # ------------------------------------------------------------------
        fig, ax = pl.subplots(figsize=(7, 4))
        ax.boxplot([rms_per_profile[:, s] for s in range(4)],
                labels=stokes_labels,
                patch_artist=True,
                boxprops=dict(facecolor='lightsteelblue'),
                medianprops=dict(color='black', linewidth=1.5))

        ax.set_title("RMS distribution per profile\n(Fast synthesis: model → z → decoder_stokes)")
        ax.set_xlabel("Stokes parameter")
        ax.set_ylabel("RMS (normalized units)")
        pl.tight_layout()
        out = os.path.join(self.output_dir, "synthesis_rms_boxplot.pdf")
        pl.savefig(out, dpi=150)
        print(f"Saved {out}")
        pl.close()

if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*.pth')
    files.sort()
    #checkpoint = files[-1]
    checkpoint = '../train/weights/2025-11-24-10_44_18_clip.pth' # (w_clip=2, w_stokes=1, w_models=2)

    deepnet = Testing(checkpoint, gpu=0, batch_size=1024)
    z_stokes, z_models, models, stokes, decoded_models, decoded_stokes = deepnet.test()

    #deepnet.plot_reconstruction(stokes, decoded_stokes, models, decoded_models, n_samples=3)

    # APPROACH 1
    #results = deepnet.fast_stokes_synthesis()
    #deepnet.plot_fast_synthesis_results(results)

    # APPROACH 2:
    synthesis_results = deepnet.fast_stokes_synthesis(models, stokes)    
    deepnet.plot_fast_synthesis_results(stokes, synthesis_results, n_samples=3)

    #for param_idx in range(6):  # loop over T, vmic, v, Bx, By, Bz
    #    deepnet.plot_tsne_1(z_stokes, z_models, models, parameter_idx=param_idx, depth_idx=40)

    # EL BUENO ES ESTE, LO QUITO PARA CORRER PRUEBAS:
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, param="T", height_idx=40, use_pca=False)
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, param="vmic", height_idx=40, use_pca=False)
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, param="v", height_idx=40, use_pca=False)
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, param="Bx", height_idx=40, use_pca=False)
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, param="By", height_idx=40, use_pca=False)
    #deepnet.plot_tsne_joint(z_stokes, z_models, models, param="Bz", height_idx=40, use_pca=False)

