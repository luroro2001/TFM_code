import numpy as np
import matplotlib.pyplot as pl # L: had to add it 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
from tqdm import tqdm
import sys
sys.path.append('../modules')
import resnet
import dataset
from collections import OrderedDict
try:
    from nvitop import Device
    NVITOP = True
except:
    NVITOP = False
import sys
import pathlib
import logging
import yaml
import argparse
from einops import rearrange

# THIS IS JUST AN ENCODER-DECODER; A VERSION OF TRAIN.PY WITHOUT CONTRASTIVE LOSS

class Training(nn.Module):
    def __init__(self, config_file):

        super().__init__()

        # Read configuration file 
        with open(config_file, 'r') as f: # L: given at the end (in main)
            self.config = yaml.safe_load(f)

        # Define the logger for output
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Check if there is a GPU available and define the computing device (CPU or GPU)
        self.cuda = torch.cuda.is_available()
        self.gpu = self.config['training']['gpu']
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        # Smoothing factor for the loss
        self.smooth = self.config['training']['smooth']
        
        # If NVITOP is installed, use it to monitor GPU usage
        if (NVITOP):
            self.handle = Device.all()[self.gpu]
            
            print("Computing in {0} : {1}".format(self.device, self.handle.name()))
        
        # Define the batch size and if decoders are used
        self.batch_size = self.config['training']['batch_size']        
        self.decoders = self.config['training']['use_decoders']
        
        ###################
        # Define the neural networks
        ###################

        # Encoder for the models
        self.encoder_models = resnet.ResidualNet(in_features=6*80,
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)
        
        # Encoder for the Stokes profiles
        self.encoder_stokes = resnet.ResidualNet(in_features=4*112, 
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)
        
        # Decoders for the models and Stokes profiles if we are using them
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
        
        # L: this creates  a scalar learnable parameter initialized to log(1/0.07)
        # in forward pass they use F.softplus(self.logit_scale) to map it to a positive value                
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))#)
        # self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        # L: prints num. of trainable parameters         
        self.logger.info('N. total parameters STOKES ENCODER : {0}'.format(sum(p.numel() for p in self.encoder_stokes.parameters() if p.requires_grad)))
        self.logger.info('N. total parameters MODELS ENCODER : {0}'.format(sum(p.numel() for p in self.encoder_models.parameters() if p.requires_grad)))

        if self.decoders:
            self.logger.info('N. total parameters STOKES DECODER : {0}'.format(sum(p.numel() for p in self.decoder_stokes.parameters() if p.requires_grad)))
            self.logger.info('N. total parameters MODELS DECODER : {0}'.format(sum(p.numel() for p in self.decoder_models.parameters() if p.requires_grad)))

        ###################
        # Define the datasets and data loaders
        ###################

        # Use four workers to load the data
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}

        # Training and validation datasets
        self.training_dataset = dataset.Dataset('stokes_training.h5', 
                                                   'models_training.h5', 
                                                   'good_profiles_training.npy',
                                                   noise=self.config['training']['noise'])
        
        self.validation_dataset = dataset.Dataset('stokes_validation.h5', 
                                                     'models_validation.h5', 
                                                     'good_profiles_validation.npy',
                                                     noise=self.config['training']['noise'])
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.training_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    **kwargs)
                
    def init_optimize(self):

        # Define the learning rate, number of epochs, weight decay and number of epochs
        self.lr = self.config['training']['lr']
        self.weight_decay = self.config['training']['weight_decay']            
        self.logger.info('Learning rate : {0}'.format(self.lr))
        self.n_epochs = self.config['training']['n_epochs']
        
        # Create the directory to save the weights if it does not exist
        p = pathlib.Path('weights/')
        p.mkdir(parents=True, exist_ok=True)

        # Define the output name for the weights using the timestamp
        current_time = time.strftime("%Y-%m-%d-%H_%M_%S")
        self.out_name = f'weights/{current_time}_clip'

        # Define the loss function
        #self.loss_fn = CLIPLoss() # not needed in this script
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)    
        
        # Define the learning rate scheduler. We use a cosine annealing scheduler        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.1*self.lr)
        
    def optimize(self):
        
        # Training and validation losses
        self.loss = []
        self.loss_val = []
        best_loss = 1e10
        
        self.logger.info('Model : {0}'.format(self.out_name))

        # Loop over the epochs
        for epoch in range(1, self.n_epochs + 1):

            # Train step for this epoch
            self.train(epoch)

            # Validation step for this epoch
            self.validate()

            # Update the learning rate
            self.scheduler.step()
            
            # Create the checkpoint dictionary
            checkpoint = {
                'epoch': epoch + 1,                
                'encoder_stokes_dict': self.encoder_stokes.state_dict(),
                'encoder_models_dict': self.encoder_models.state_dict(),
                'config': self.config,                
                'best_loss': best_loss,
                'loss': self.loss,
                'loss_val': self.loss_val,
                'optimizer': self.optimizer.state_dict(),
            }

            # If decoders are used, add them to the checkpoint
            if (self.decoders):
                checkpoint['decoder_stokes_dict'] = self.decoder_stokes.state_dict()
                checkpoint['decoder_models_dict'] = self.decoder_models.state_dict()

            # If a scheduler is used, add it to the checkpoint
            if (self.scheduler is not None):
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            # Save the current checkpoint
            self.logger.info(f'Saving model {self.out_name}.pth')
            torch.save(checkpoint, f'{self.out_name}.pth')

            # If the validation loss is the best until now, save the checkpoint as the best model
            if (self.loss_val[-1] < best_loss):
                self.logger.info(f"Saving model {self.out_name}.pth.best")     
                best_loss = self.loss_val[-1]
                torch.save(checkpoint, f'{self.out_name}.pth.best')

            # Update the best loss
            best_loss = min(self.loss_val[-1], best_loss)
            
        
    def train(self, epoch):
        # Put models in training mode
        self.encoder_stokes.train()
        self.encoder_models.train()
        if (self.decoders):
            self.decoder_stokes.train()
            self.decoder_models.train()
        
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        
        # Iterator for the training data with a progress bar
        t = tqdm(self.train_loader)
        
        # Current learning rate
        current_lr = self.scheduler.get_last_lr()[0]

        # Loop over the batches    
        for batch_idx, (stokes, models) in enumerate(t):

            # Move data to the computing device (GPU or CPU)
            models = models.to(self.device)
            stokes = stokes.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()

            stokes_flat = rearrange(stokes, 'b c h -> b (c h)')
            models_flat = rearrange(models, 'b c h -> b (c h)')
                        
            # Use encoder to get z_stokes and z_models
            z_stokes = self.encoder_stokes(stokes_flat)
            z_models = self.encoder_models(models_flat)

            # Normalize the latent vectors
            #z_stokes = F.normalize(z_stokes, dim=-1)
            #z_models = F.normalize(z_models, dim=-1)

            # Compute contrastive loss
            #loss_clip = self.loss_fn(z_stokes, z_models, logit_scale=F.softplus(self.logit_scale))
            
            # If decoders are used, decode and compute reconstruction loss. We use MSE loss
            if self.decoders:
                decoded_stokes = self.decoder_stokes(z_stokes)
                decoded_models = self.decoder_models(z_models)
            
                loss_stokes = F.mse_loss(decoded_stokes, stokes_flat)
                loss_models = F.mse_loss(decoded_models, models_flat)
            else:
                loss_stokes = torch.tensor(0.0).to(self.device)
                loss_models = torch.tensor(0.0).to(self.device)

            # Total loss (only reconstruction, no CLIP)
            loss = loss_stokes + loss_models

            # Backpropagation                    
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Now do some output for the user
            # Compute the smoothed losses
            if (batch_idx == 0):
                loss_avg = loss.item()
                #loss_clip_avg = loss_clip.item()
                loss_stokes_avg = loss_stokes.item()
                loss_models_avg = loss_models.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                #loss_clip_avg = self.smooth * loss_clip.item() + (1.0 - self.smooth) * loss_clip_avg
                loss_stokes_avg = self.smooth * loss_stokes.item() + (1.0 - self.smooth) * loss_stokes_avg
                loss_models_avg = self.smooth * loss_models.item() + (1.0 - self.smooth) * loss_models_avg
            
            # If NVITOP is installed, get the GPU usage
            if (NVITOP):
                gpu_usage = f'{self.handle.gpu_utilization()}'                
                memory_usage = f' {self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
            else:
                gpu_usage = 'NA'
                memory_usage = 'NA'

            # Update the progress bar
            tmp = OrderedDict()
            tmp['gpu'] = f'{gpu_usage}'
            tmp['mem'] = f'{memory_usage}'
            tmp['lr'] = f'{current_lr:8.6f}'
            #tmp['scale'] = f'{F.softplus(self.logit_scale):8.6f}'
            #tmp['L_c'] = f'{loss_clip_avg:8.6f}'
            tmp['L_s'] = f'{loss_stokes_avg:8.6f}'
            tmp['L_m'] = f'{loss_models_avg:8.6f}'
            tmp['L'] = f'{loss_avg:8.6f}'
            t.set_postfix(ordered_dict = tmp)

            # Save the smoothed loss
            self.loss.append(loss_avg)
                    
        return

    def validate(self):
        # L: put models in evaluation mode 
        self.encoder_stokes.eval()
        self.encoder_models.eval()
        if (self.decoders):
            self.decoder_stokes.eval()
            self.decoder_models.eval()
        
        t = tqdm(self.validation_loader)

        with torch.no_grad():
            for batch_idx, (stokes, models) in enumerate(t):
                models = models.to(self.device)
                stokes = stokes.to(self.device)

                stokes_flat = rearrange(stokes, 'b c h -> b (c h)')
                models_flat = rearrange(models, 'b c h -> b (c h)')

                z_stokes = self.encoder_stokes(stokes_flat)
                z_models = self.encoder_models(models_flat)

                #z_stokes = F.normalize(z_stokes, dim=-1)
                #z_models = F.normalize(z_models, dim=-1)

                #loss_clip = self.loss_fn(z_stokes, z_models, logit_scale=F.softplus(self.logit_scale))

                if self.decoders:
                    decoded_stokes = self.decoder_stokes(z_stokes)                
                    decoded_models = self.decoder_models(z_models)                
                                
                    loss_stokes = F.mse_loss(decoded_stokes, stokes_flat)
                    loss_models = F.mse_loss(decoded_models, models_flat)
                else:
                    loss_stokes = torch.tensor(0.0).to(self.device)
                    loss_models = torch.tensor(0.0).to(self.device)

                loss = loss_stokes + loss_models
                                                                
                if (batch_idx == 0):
                    loss_avg = loss.item()
                    #loss_clip_avg = loss_clip.item()
                    loss_stokes_avg = loss_stokes.item()
                    loss_models_avg = loss_models.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                    #loss_clip_avg = self.smooth * loss_clip.item() + (1.0 - self.smooth) * loss_clip_avg
                    loss_stokes_avg = self.smooth * loss_stokes.item() + (1.0 - self.smooth) * loss_stokes_avg
                    loss_models_avg = self.smooth * loss_models.item() + (1.0 - self.smooth) * loss_models_avg

                if (NVITOP):
                    gpu_usage = f'{self.handle.gpu_utilization()}'                
                    memory_usage = f' {self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
                else:
                    gpu_usage = 'NA'
                    memory_usage = 'NA'

                tmp = OrderedDict()
                tmp['gpu'] = f'{gpu_usage}'
                tmp['mem'] = f'{memory_usage}'
                #tmp['L_c'] = f'{loss_clip_avg:8.6f}'
                tmp['L_s'] = f'{loss_stokes_avg:8.6f}'
                tmp['L_m'] = f'{loss_models_avg:8.6f}'
                tmp['L'] = f'{loss_avg:8.6f}'
                
                t.set_postfix(ordered_dict = tmp)

                self.loss_val.append(loss_avg)
            
        return

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser("parallel")

    parser.add_argument(
        '--config',
        type=str,
        default='conf.yaml',
        help='Path to the configuration file'
    )

    args = parser.parse_args()

    deepnet = Training(args.config)

    deepnet.init_optimize()
    deepnet.optimize()