import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio
import sys
import os
from torchmetrics.functional import structural_similarity_index_measure as SSIM 
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import mean_absolute_error as MAE
from scripts.Wavelet import WaveletLoss  # Assuming WaveletLoss is defined in Wavelet.py


sys.path.append(os.path.join(sys.path[0], '../..'))
from src.network.VNet import VNet  

class VNetModel(pl.LightningModule):
    def __init__(self, params):
        super(VNetModel, self).__init__()

        self.save_hyperparameters()
        self.loss_type = params.model.loss_type
        #use hybrid loss with WaveletLoss MAE + WaveletLoss
        allowed_loss = ['MAE', 'Wavelet', 'Hybrid']  # Added 'Wavelet'
        if self.loss_type == 'Wavelet':
            self.criterion_wavelet = WaveletLoss(wavelet='db4', levels=3, loss_type='l1')   
        else:
            self.criterion_wavelet = None
        # Check if the loss type is allowed
        # If not, raise an error with a clear message
        if self.loss_type == 'Wavelet':
            print("Using Wavelet Loss")
        else:
            print("Using MAE Loss")

        error_msg = f"Allowed loss types: {allowed_loss}"
        assert self.loss_type in allowed_loss, error_msg


        # Hyperparameters     
        self.lr = params.model.learning_rate
        self.l2 = params.model.weight_decay 

        self.bn = params.data.batch_size

        self.MAE_alpha = params.model.MAE_loss_weight
        self.percep_alpha = params.model.preceptual_loss_weight
        self.wavelet_alpha = params.model.get("wavelet_loss_weight", 1.0)  # or hardcode 1.0

        self.criterion_MAE = nn.L1Loss()   
        self.criterion_MSE = nn.MSELoss()
        self.criterion_Wavelet = WaveletLoss(wavelet='db4', levels=3, loss_type='l1')

        print("Initializing VNetModel with params:", params)  # Debug logging
        self.basemodel = VNet()

    def forward(self, x):
  
        
        # x = self.conv1(x)
        # print(f"After conv1 shape: {x.shape}")  # Debug
        # x = self.bn1(x)
        # print(f"After bn1 shape: {x.shape}")  # Debug
        # x = self.relu1(x)  # This is where it fails
       
        #adding check for input shape
        # x: [B,C,D,H,W] (batch size, channels, depth, height, width)
        assert x.ndim == 5, f"Expected [B,C,D,H,W], got {x.shape}" 
        assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"


        # make x x= self.VNet(x) 
        # x = self.VNet(x)

        x=self.basemodel(x)  # Call the VNet model
        #print(f"Output shape: {x.shape}")
        
        return x  # Call the VNet model


    #optimizer Config 
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.l2)
        return {"optimizer": optimizer}
    
    def on_train_epoch_end(self):
        # Example: log something manually
        epoch = self.current_epoch
        self.log("epoch_end_log", epoch, prog_bar=True)

    #Training loop :
    def training_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA]
        img_7t = batch['t1_7T'][tio.DATA]
        
        img_s7t = self.forward(img_3t)

        # MAE loss
        mae_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())
        content_loss = self.MAE_alpha * mae_loss

        # Wavelet loss
        wavelet_loss = self.criterion_Wavelet(img_s7t.squeeze(), img_7t.squeeze())
        wavelet_loss = self.wavelet_alpha * wavelet_loss

        # Combine losses
        if self.loss_type == 'Hybrid':
            # Hybrid loss: MAE + Wavelet    
            total_loss = content_loss + wavelet_loss
            self.log('train_Hybrid_loss', total_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
            self.log('train_MAE_component', mae_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
            self.log('train_Wavelet_component', wavelet_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
        elif self.loss_type == 'Wavelet':
            # Only Wavelet loss
            total_loss = wavelet_loss
            self.log('train_Wavelet_loss', total_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
        elif self.loss_type == 'MAE':
            # Only MAE loss
            total_loss = content_loss
            self.log('train_MAE_loss', total_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return total_loss



#Validation setup
    def validation_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA]
        img_7t = batch['t1_7T'][tio.DATA]
        img_s7t = self.forward(img_3t)

        # MAE loss
        mae_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())
        content_loss = self.MAE_alpha * mae_loss

        # Wavelet loss
        wavelet_loss = self.criterion_Wavelet(img_s7t.squeeze(), img_7t.squeeze())
        wavelet_loss = self.wavelet_alpha * wavelet_loss

        # Combine losses
        if self.loss_type == 'Hybrid':
            # Hybrid loss: MAE + Wavelet    
            total_loss = content_loss + wavelet_loss
            self.log('val_Hybrid_loss', total_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
            self.log('val_MAE', mae_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
            self.log('val_Wavelet_component', wavelet_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
        elif self.loss_type == 'Wavelet':
            # Only Wavelet loss
            total_loss = wavelet_loss
            self.log('val_Wavelet_loss', total_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
        elif self.loss_type == 'MAE':
            # Only MAE loss
            total_loss = content_loss
            self.log('val_MAE', total_loss, on_step=True, on_epoch=True,
                     logger=True, batch_size=self.bn, sync_dist=True)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return total_loss

    