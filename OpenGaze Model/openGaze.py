import torch
import torch.nn as nn
import pytorch_lightning as pl
from openGazeData import openGazeData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import os
'''
This code defines a comprehensive model for gaze tracking, leveraging
both eye images and facial landmarks to predict the gaze direction.
'''
class openGaze(pl.LightningModule):
    def __init__(self, data_path, save_path):
        super(openGaze, self).__init__()   

        self.learningRate = 0.016
        self.batch_size = 256
        self.data_path = data_path
        self.workers = 7
        print("Data Path: ", data_path)
        self.save_path = save_path
    
        self.eyeModel = eyeModel() #Initialized the eye model
        self.landMark = landMark() #Initialized landmark model

        #a sequential model to combine features from both eyes and landmarks and predict gaze direction
        self.combined_model = nn.Sequential(nn.Linear(512+512+16, 8),
                                            nn.BatchNorm1d(8, momentum=0.9),
                                            nn.Dropout(0.12),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(8, 4),
                                            nn.BatchNorm1d(4, momentum=0.9),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(4, 2),)  
        
    def forward(self, leftEye, rightEye, lms):
        #Flatten features of left and right eye
        l_eye_feat = torch.flatten(self.eyeModel(leftEye), 1)
        r_eye_feat = torch.flatten(self.eyeModel(rightEye), 1)
        
        lm_feat = self.landMark(lms) 
        
        #Concatenate the flattened features
        combined_feat = torch.cat((l_eye_feat, r_eye_feat, lm_feat), 1)
        out = self.combined_model(combined_feat) #Final output 
        return out
    
    def training_step(self, batch, batch_idx):
        _, l_eye, r_eye, kps, y, _, _ = batch
        y_hat = self(l_eye, r_eye, kps)
        loss = F.mse_loss(y_hat, y)
        print('train_loss', loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def train_dataloader(self):
        train_dataset = openGazeData(self.data_path+"/train/", phase='train')
        print(self.data_path+ "/train/")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True, persistent_workers=True)
        print('Num_train_files', len(train_dataset))
        return train_loader
    
    def val_dataloader(self):
        dataVal = openGazeData(self.data_path+"/val/", phase='val')
        val_loader = DataLoader(dataVal, batch_size=self.batch_size, num_workers=self.workers, shuffle=False, persistent_workers=True)
        print('Num_val_files', len(dataVal))
        self.logger.log_hyperparams({'Num_val_files': len(dataVal)})

        return val_loader
    
    def validation_step(self, batch, batch_idx):
        _, l_eye, r_eye, kps, y, _, _ = batch
        y_hat = self(l_eye, r_eye, kps)
        val_loss = F.mse_loss(y_hat, y)
        print('val_loss', val_loss)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learningRate, betas=(0.9, 0.999), eps=1e-07, weight_decay=1e-4)
    
    # Implement learning rate scheduler with warm-up and ReduceLROnPlateau
        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            return 1
        
        scheduler_warmup = LambdaLR(optimizer, lr_lambda)
        scheduler_plateau = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler_plateau,
                'monitor': 'val_loss'
            },
            'scheduler_warmup': scheduler_warmup
        }
'''
This class defines a convolutional neural network (CNN) model for processing eye images.
It consists of three convolutional layers, each followed by batch normalization, a leaky
ReLU activation, average pooling, and dropout for regularization.
'''
class eyeModel(nn.Module): 
    def __init__(self):
        super(eyeModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=0), #First Convolutional Layer
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.CrossMapLRN2d(size=5, alpha=0.00001, beta=0.75, k = 1.0),
            nn.Dropout(0.02),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), #Second Convolutional Layer
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.CrossMapLRN2d(size=5, alpha=0.00001, beta=0.75, k = 1.0),
            nn.Dropout(0.02),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), #Third Convolutional Layer
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.CrossMapLRN2d(size=5, alpha=0.00001, beta=0.75, k = 1.0),
            nn.Dropout(0.02),
        )

    def forward(self, x):
        x = self.model(x)
        return x

'''
This class defines a model for processing facial landmarks.
It uses fully connected layers with ReLU activations and batch normalization.
'''   
class landMark(nn.Module):
    def __init__(self):
        super(landMark, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 128),  #First fully connected Layer
            nn.BatchNorm1d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16), #Second fully connected Layer
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16), #Third fully connected Layer
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
#xy = openGaze()
#print(xy.eval())