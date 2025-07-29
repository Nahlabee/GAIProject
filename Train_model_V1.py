### Imports
import torch
import pytorch_lightning as pl
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import torchmetrics
from box import Box #box.Box for accessing dict keys like attributes.......... 
from datetime import datetime #import time for time stamping 


#import metrics-----------------------------------------------------------------------------------------------------------------
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import mean_absolute_error as MAE
#import wavelet metrics if needed

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS # LPIPS metric for perceptual loss to add to training loop 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
import sys, os, json

### Adding parent directory to path---------------------------------------------------------------------------------------------
sys.path.append(os.path.join(sys.path[0], '../..')) # go back two directories to access src folder
#go back three directories to access src folder

### Importing modules------------------------------------------------------------------------------------------------------------- 
from src.dataloader.patch_dataloader import patch_dataloader
from src.dataloader.data_utils_train import load_onefold_dataset, load_all
from src.network.VNet_model_HybLoss import VNetModel  # Importing the VNetModel with hybrid loss
from src.network.VNet_model import VNetModel
from src.network.WatNet_model import WatNet3DModel, WatNet2DModel


### Set working directory to script directory
script_dir = os.path.dirname(os.path.realpath(__file__)) #gets the directory of the current script
current_dir = os.getcwd()  #working dir should be /home/nbelloula/GAI/Synthetic7TMRI4 that contains the src folder

os.chdir(script_dir)
print(f"Current working directory: {current_dir}")
print(f"Script directory: {script_dir}")
print("Starting training pipeline...")

### Utility Function to Load Parameters with Relative Path
def load_params(param_filename):
    """
    Load parameters from JSON file using relative path.
    Config file should be in ../../config/ relative to this script.

    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Direct path to config file (up 2 levels, then into config directory)
    param_path = os.path.join(script_dir, '..','..', 'config', param_filename)
    param_path = os.path.abspath(param_path)  # Convert to absolute path
    print(param_path)
    if os.path.exists(param_path):
        print(f"Found config file at: {param_path}")
        with open(param_path, 'r') as json_file:
            params = json.load(json_file)
        return Box(params)
    else:
        raise FileNotFoundError(f"Could not find {param_filename} at expected location: {param_path}")

### Custom Callback for Loss Tracking and Plotting
class LossTracker(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_epochs = []
        self.val_epochs = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Get training loss
        train_loss = trainer.callback_metrics.get('train_MAE', None)
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
            self.train_epochs.append(trainer.current_epoch)
            
    def on_validation_epoch_end(self, trainer, pl_module): 
        # Get validation loss
        val_loss = trainer.callback_metrics.get('val_MAE', None)
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())
            self.val_epochs.append(trainer.current_epoch)
        
        # Plot every 10 epochs or at the end
        if (trainer.current_epoch + 1) % 10 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            self.plot_losses()
            
    def plot_losses(self):
        """Plot and save training and validation losses"""
        if not self.train_losses and not self.val_losses:
            print("No loss data to plot yet.")
            return
            
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Combined train and validation loss
        if self.train_losses and self.train_epochs:
            ax1.plot(self.train_epochs, self.train_losses, 'b-', label='Training MAE', linewidth=2)
        if self.val_losses and self.val_epochs:
            ax1.plot(self.val_epochs, self.val_losses, 'r-', label='Validation MAE', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE Loss')
        ax1.set_title('Training and Validation MAE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training loss only (more detailed view)
        if self.train_losses and self.train_epochs:
            ax2.plot(self.train_epochs, self.train_losses, 'b-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Training MAE Loss')
            ax2.set_title('Training MAE Loss (Detailed)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        current_epoch = max(self.train_epochs) if self.train_epochs else max(self.val_epochs) if self.val_epochs else 0
        plot_path = os.path.join(self.save_dir, f'loss_plot_epoch_{current_epoch}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to: {plot_path}")
        
    def save_loss_data(self):
        """Save loss data to files for later analysis"""
        # Save as numpy arrays
        if self.train_losses:
            np.save(os.path.join(self.save_dir, 'train_losses.npy'), np.array(self.train_losses))
            np.save(os.path.join(self.save_dir, 'train_epochs.npy'), np.array(self.train_epochs))
        
        if self.val_losses:
            np.save(os.path.join(self.save_dir, 'val_losses.npy'), np.array(self.val_losses))
            np.save(os.path.join(self.save_dir, 'val_epochs.npy'), np.array(self.val_epochs))
        
        # Save as text file for easy reading
        with open(os.path.join(self.save_dir, 'loss_log.txt'), 'w') as f:
            f.write("Epoch\tTrain_MAE\tVal_MAE\n")
            
            # Create a combined view of all epochs
            all_epochs = sorted(set(self.train_epochs + self.val_epochs))
            
            for epoch in all_epochs:
                # Find training loss for this epoch
                train_loss = 'N/A'
                if epoch in self.train_epochs:
                    train_idx = self.train_epochs.index(epoch)
                    train_loss = self.train_losses[train_idx]
                
                # Find validation loss for this epoch
                val_loss = 'N/A'
                if epoch in self.val_epochs:
                    val_idx = self.val_epochs.index(epoch)
                    val_loss = self.val_losses[val_idx]
                
                f.write(f"{epoch}\t{train_loss}\t{val_loss}\n")

### Load configuration parameters - Using direct relative path
try:
    params = load_params('params_VNet_HybLoss.json')
    # params = load_params('params_WatNet2D.json')  # Uncomment for WatNet2D
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure your config file is at: ../../config/params_VNet_HybLoss.json relative to this script.")
    sys.exit(1)

torch.set_float32_matmul_precision(params.training.matmul_precision)

# Determine if ClearML is enabled
USE_CLEARML = params.get("logging", {}).get("use_clearml", False)
if USE_CLEARML:
    from clearml import Task

# Check for supported model names
assert params.model.name in ["VNet", "WatNet2D", "WatNet3D"], \
    "Model name should be VNet, WatNet2D, or WatNet3D"

### Fold Handling
if params.data.fold_ind == "allfolds":
    fold_inds = np.arange(params.data.kfold_num)
elif params.data.fold_ind == "finalmodel":
    fold_inds = None
else:
    fold_inds = params.data.fold_ind

# Training Setup---------------------------------------------------------------------------------------------------------------

# get experiment name for clearml
exp_name = params.training.exp_name
main_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define a callback to track epochs with a timestamp
class EpochTracker(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Print the current epoch with a timestamp
        print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

###-----------------------------------------------------------------------------------------------------------------------------
### Training Loop --------------------------------------------------------------------------------------------------------------
###-----------------------------------------------------------------------------------------------------------------------------

base_output_dir = "/home/nbelloula/GAI/Synthetic7TMRI4/Outputs"
os.makedirs(base_output_dir, exist_ok=True)
main_exp_dir = os.path.join(base_output_dir, f"{exp_name}_{main_timestamp}")
checkpoint_dir = os.path.join(main_exp_dir, 'Checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

if fold_inds is not None: # the case where we train on multiple folds

    for fold_ind in fold_inds:
        exp_ver = f"MAEloss_fold{fold_ind}_{main_timestamp}"
        fold_checkpoint_dir = os.path.join(checkpoint_dir, exp_ver)
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        
        if USE_CLEARML:
            task = Task.init(project_name=exp_name, task_name=exp_ver)
            task.set_parameters_as_dict(params.to_dict())

        train_dataset, val_dataset = load_onefold_dataset(params, fold_ind=fold_ind)
        train_loader, _ = patch_dataloader(train_dataset, params)
        val_loader, aggregator = patch_dataloader(val_dataset, params)

        if params.model.name == 'VNet':
            model = VNetModel(params)
        elif params.model.name == 'WatNet2D':
            model = WatNet2DModel(params)
        elif params.model.name == 'WatNet3D':
            model = WatNet3DModel(params)
        
        # Create plots directory: /home/nbelloula/GAI/Synthetic7TMRI4/Outputs/exp_name_timestamp/Plots/MAEloss_fold{X}_{timestamp}/
        plots_dir = os.path.join(main_exp_dir, 'Plots', exp_ver)
        logger = TensorBoardLogger('tensorlog', name=exp_name, version=exp_ver)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Initialize loss tracker
        loss_tracker = LossTracker(plots_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_checkpoint_dir,
            save_last=True,
            every_n_epochs=params.checkpoint.save_frequency_epoch,
            save_top_k=params.checkpoint.save_topk,
            monitor=params.checkpoint.monitor,
            filename='{epoch}-{step}-{val_MAE:.8f}'
        )

        # Add Early Stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_MAE',  # Monitor validation MAE already defined in params but yeah
            patience=5,         # Number of epochs with no improvement after which training will be stopped; might need tuning!!
            mode='min',         # For MAE, we want to minimize (lower is better)
            verbose=True,       # Print message when early stopping triggers
            min_delta=0.0001,   # Minimum change to qualify as an improvement
            check_finite=True,  # Stop if monitored metric becomes NaN or infinite
            strict=True,        # Whether to crash if monitor is not found
            check_on_train_epoch_end=None,  # Check at end of validation epoch (default)
            log_rank_zero_only=False,  # Log on all ranks in DDP
        )

        trainer = pl.Trainer(
            max_epochs=params.training.num_epochs,
            accelerator='auto',
            strategy='auto', #auto will use DDP if multiple GPUs are available
            logger=logger,
            precision=params.training.precision,
            log_every_n_steps=params.training.log_every_n_steps,
            callbacks=[lr_monitor, checkpoint_callback, RichProgressBar(), EpochTracker(), early_stopping_callback, loss_tracker]
        )
        
        print(f"Starting training: {exp_name} - {exp_ver}")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
        print(f"Plots will be saved to: {plots_dir}")
        trainer.fit(model, train_loader, val_loader)
        
        # Save final loss data and create final plot
        loss_tracker.save_loss_data()
        loss_tracker.plot_losses()
        
        print(f"Finished training: {exp_name} - {exp_ver}")
        print(f"Loss plots and data saved to: {plots_dir}")
        
        if USE_CLEARML:
            task.close()

else:
    exp_ver = f"MAEloss_finalmodel_{main_timestamp}"
    fold_checkpoint_dir = os.path.join(checkpoint_dir, exp_ver)
    os.makedirs(fold_checkpoint_dir, exist_ok=True)

    if USE_CLEARML:
        task = Task.init(project_name=exp_name, task_name=exp_ver)
        task.set_parameters_as_dict(params.to_dict())

    train_dataset = load_all(params)
    train_loader, aggregator = patch_dataloader(train_dataset, params)

    if params.model.name == 'VNet':
        model = VNetModel(params)
    elif params.model.name == 'WatNet2D':
        model = WatNet2DModel(params)
    elif params.model.name == 'WatNet3D':
        model = WatNet3DModel(params)

     
    # Create plots directory: /home/nbelloula/GAI/Synthetic7TMRI4/Outputs/exp_name_timestamp/Plots/MAEloss_finalmodel_{timestamp}/
    plots_dir = os.path.join(main_exp_dir, 'Plots', exp_ver)
    
    logger = TensorBoardLogger('tensorlog', name=exp_name, version=exp_ver)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Initialize loss tracker (for final model, only training loss)
    loss_tracker = LossTracker(plots_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=fold_checkpoint_dir,
        save_last=True,
        every_n_epochs=params.checkpoint.save_frequency_epoch,
        save_top_k=params.checkpoint.save_topk,
        monitor="train_MAE",
        filename='{epoch}-{step}-{train_MAE:.8f}-' + main_timestamp  # Fixed: use main_timestamp instead of undefined timestamp
    )

    trainer = pl.Trainer(
        max_epochs=params.training.num_epochs,
        accelerator='auto',
        strategy='auto',
        logger=logger,
        precision=params.training.precision,
        log_every_n_steps=params.training.log_every_n_steps,
        callbacks=[lr_monitor, checkpoint_callback, RichProgressBar(), EpochTracker(), loss_tracker]
        # Add early_stopping_callback to the list above if you want to use it for final model training
    )

    print(f"Starting training: {exp_name} - {exp_ver}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Plots will be saved to: {plots_dir}")
    
    # Note: Only training data is used for final model (no validation)
    trainer.fit(model, train_loader)
    
    # Save final loss data and create final plot
    loss_tracker.save_loss_data()
    loss_tracker.plot_losses()
    
    print(f"Finished training: {exp_name} - {exp_ver}")
    print(f"Loss plots and data saved to: {plots_dir}")
    
    if USE_CLEARML:
        task.close()


