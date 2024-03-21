import torchvision.models as models
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import pytorch_lightning as pl

class GRN(pl.LightningModule):
  def __init__(self, data_path, save_path, use_rnn=True):
    super(GRN, self).__init__()
    
    #For training and testing
    self.learningRate = 0.016
    self.batch_size = 256
    self.data_path = data_path
    self.workers = 1
    print("Data Path: ", data_path)
    self.save_path = save_path
    
    #Initializing EyeNet and GazeRefineNet - to be defined in forward
    self.eyeNet = EyeNet() #Initialized the eye model
    self.gazeRefineNet = gazeRefineNet() #Initialized landmark model
    
  #Need to be refined after the entire model is done
  def forward(self, leftEye, rightEye):
        #Flatten features of left and right eye 
        l_eye_feat = torch.flatten(self.eyeNet(leftEye), 1) #Inputs need to be given as per the eyenet model
        r_eye_feat = torch.flatten(self.eyeNet(rightEye), 1)
        
        grn_out = self.gazeRefineNet(l_eye_feat,r_eye_feat)  #Inputs need to be given as per the gazerefinenet model
        
        return grn_out
    
  #Need to be refined later
  def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        preds = ...
        self.training_step_outputs.append(preds)
        return loss
    
  def validation_step(self, batch, batch_idx):
        x, y = batch
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        pred = ...
        self.validation_step_outputs.append(pred)
        return pred
  def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, betas=(0.9, 0.999), eps=1e-07)
#         scheduler = ExponentialLR(optimizer, gamma=0.64, verbose=True)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

#EyeNet model
class EyeNet(nn.Module):
  def __init__(self, use_rnn=True):
    super(EyeNet, self).__init__()
    self.resnet = models.resnet18(
        #block = models.resnet.BasicBlock,
        #layers = [2,2,2,2],
        pretrained=False,        # Not using pre-trained weights
        num_classes=128,        # (IMP)Number of output classes - needs to be defined based on number of eye features needed
        norm_layer=nn.InstanceNorm2d,  # Normalization layer - layers need to be added
    )
    self.use_rnn = use_rnn

    # Optional recurrent component - GRUCell
    if use_rnn:
            self.rnn = nn.GRUCell(input_size=128, hidden_size=128)
            #self.fc_gaze = nn.Linear(128, 3)  # Output size for gaze direction

    self.fc_gaze = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.SELU(inplace=True),
                    nn.Linear(128, 2, bias=False),
                    nn.Tanh(),
                )  # Output size for gaze direction
    self.fc_pupil = nn.Sequential(
                      nn.Linear(128, 128),
                      nn.SELU(inplace=True),
                      nn.Linear(128, 1),
                      nn.ReLU(inplace=True),
                  )  # Output size for pupil size
    
  def forward(self, input_eye_image, rnn_output=None):

    features = self.resnet(input_eye_image)

    if self.use_rnn:
      rnn_features=features
      batch_size, feature_size = features.shape
      hidden = torch.zeros(batch_size, 128, device=rnn_features.device)
      previous_results = []
      output=[]

      for i in range(batch_size):  # Loop through layers

        if rnn_output is not None:
            previous_results = output[i-1] if i>0 else None

        GRUResult= self.rnn(rnn_features,hidden)

        if isinstance(GRUResult, tuple):
          rnn_features=GRUResult[0]
          output[i] = GRUResult
        else:
          rnn_features = GRUResult
          output.append(GRUResult)
          
      features=rnn_features
      
    #to calculate point of gaze
    gaze_direction = (0.5 * np.pi) * self.fc_gaze(features)
    gaze_direction_vector= convert_angles_to_vector(gaze_direction)
    point_of_gaze_mm = calculate_intersection_with_screen(origin,gaze_direction_vector)
    point_of_gaze_px = mm_to_pixels(point_of_gaze_mm,screen_size_mm, screen_size_pixels) # need to get from screen.json
    pupil_size =self.fc_pupil(features)
    print("Gaze Direction shape before linear layer:", gaze_direction.shape)
    print("Pupil Size shape before linear layer:", pupil_size.shape)
    return gaze_direction, pupil_size, point_of_gaze_px

#Converting pitch and yaw to a vector - to convert gaze direction to a vector

def convert_angles_to_vector(angles):
    # Check if the input angles are 2-dimensional (pitch and yaw)
    if angles.shape[1] == 2:
        sine_values = torch.sin(angles)
        cosine_values = torch.cos(angles)
        # Construct and return the direction vector
        return torch.stack([cosine_values[:, 0] * sine_values[:, 1], sine_values[:, 0], cosine_values[:, 0] * cosine_values[:, 1]], dim=1)
    # Normalize the vector if the input is 3-dimensional
    elif angles.shape[1] == 3:
        return F.normalize(angles, dim=1)
    else:
        # Raise an error for unsupported input dimensions
        raise ValueError(f'Unexpected input dimensions: {angles.shape}')

# To calculate point of gaze, gaze origin assumed to be 0,0,0      
def calculate_intersection_with_screen(o, direction):
    # Assuming o = (0, 0, 0) for simplicity
    # Solve for t when z = 0
    t = -o[:, 2] / direction[:, 2]
    
    # Calculate intersection point in millimeters
    p_x = o[:, 0] + t * direction[:, 0]
    p_y = o[:, 1] + t * direction[:, 1]
    
    return torch.stack([p_x, p_y], dim=-1)

def mm_to_pixels(intersection_mm, screen_size_mm, screen_size_pixels):
    # Unpack screen dimensions
    screen_width_mm, screen_height_mm = screen_size_mm
    screen_width_px, screen_height_px = screen_size_pixels
    
    # Calculate pixels per millimeter
    ppmm_x = screen_width_px / screen_width_mm
    ppmm_y = screen_height_px / screen_height_mm
    
    # Convert intersection point from mm to pixels
    intersection_px = intersection_mm * torch.tensor([ppmm_x, ppmm_y])
    return intersection_px
  
  
#GazeRefineNet model

class GazeRefineNet(nn.Module):
    def __init__(self, in_shape, out_shape, activation=nn.ReLU):
        super(GazeRefineNet, self).__init__()
        # Dynamically set input channels based on configuration
        in_channels = 4 if config['load_screen_content'] else 1
        self.do_skip = config['use_skip_connections']

        # Define initial convolution layers to process the input image
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Example of a simplified backbone architecture
        self.backbone = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Additional layers would be added here...
        )

        # Final convolution to generate the heatmap
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Assuming the output is a heatmap
        )

    def forward(self, input_dict):
        # Example preprocessing step
        # Assuming input_dict contains 'screen_frame' and 'heatmap_initial'
        if config['load_screen_content']:
            input_image = torch.cat([input_dict['screen_frame'], input_dict['heatmap_initial']], dim=1)
        else:
            input_image = input_dict['heatmap_initial']

        # Pass through initial convolutions
        x = self.initial_conv(input_image)
        # Pass through the backbone
        x = self.backbone(x)
        # Generate final heatmap
        final_heatmap = self.final_conv(x)

        return final_heatmap

# Need to change configuration accordingly
config = {
    'load_screen_content': True,
    'use_skip_connections': True,
}

# Initialize the model
model = GazeRefineNet(config)