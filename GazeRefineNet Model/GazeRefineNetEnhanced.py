import torchvision.models as models
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import os
import json
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import scipy.ndimage

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import torch.nn as nn

from GazeRefineNet.gazeRefineNetData import gazeRefineNetData

#For heatmap generation
sigma=10

class GRN(pl.LightningModule):
  def __init__(self, data_path, save_path):
    print("grn init")
    screen_size_mm = [123.8 , 53.7]
    screen_size_pixels = [320,568]
    screen_size_pixels_heatmap = [640,1136]
    sigma=10
    torch.set_float32_matmul_precision('medium')
    super(GRN, self).__init__()
    
    #For training and testing
    self.learningRate = 0.016
    self.batch_size = 1
    self.data_path = data_path
    self.workers = 1
    print("Data Path: ", data_path)
    self.save_path = save_path
    self.image_dir = r'C:\Program Files\Common Files\ProDataset\train\images\cropped_eyes'
    self.meta_dir = r'C:\Program Files\Common Files\ProDataset\train\meta'
        
    self.gazeRefineNet = GazeRefineNet() #Initialized landmark model
    
 
  def forward(self, kps,screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm, initial_heatmap=None):
        average_gaze_direction = self.gazeRefineNet(kps,screen_w, screen_h, img_tensor_l, lx, ly, img_tensor_r, rx, ry, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm)
        print("return forward grn")
        return average_gaze_direction
    
  def training_step(self, batch, batch_idx):
        _, kps, out, screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm = batch
        grn_out = self.forward(kps,screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r,attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm)
        loss = F.mse_loss(grn_out, out)
        print('train_loss', loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
  
  def collate_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:  # If all items were None
        return None
    return default_collate(batch)
  
  def train_dataloader(self):
        print("inside train loader")
        train_dataset = gazeRefineNetData(self.image_dir, self.meta_dir)
        print("inside train loader", train_dataset)
        print(self.data_path+ "/train/")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True, collate_fn=self.collate_skip_none, persistent_workers=True)
        print('Num_train_files', len(train_dataset))
        return train_loader
  
  def val_dataloader(self):
        print("inside val loader")
        val_data_path = r'C:\Program Files\Common Files\ProDataset\val\images\cropped_eyes'
        val_meta_path = r'C:\Program Files\Common Files\ProDataset\val\meta'
        train_dataset = gazeRefineNetData(val_data_path, val_meta_path)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True, persistent_workers=True)
        print('Num_train_files', len(train_dataset))
        return train_loader
    
  def validation_step(self, batch, batch_idx):
        _, kps, out, screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm  = batch
        grn_out = self.forward(kps,screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm )
        loss = F.mse_loss(grn_out, out)
        print('val_loss', loss)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return loss
  
  def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
                        'monitor': 'train_loss',  # Note how the 'monitor' key is part of the dictionary.
                        'interval': 'epoch',
                        'frequency': 1,
                        'reduce_on_plateau': True,  # This line is actually unnecessary as it's implicit with ReduceLROnPlateau
                        }
        return [optimizer], [lr_scheduler]
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

#EyeNet model
class EyeNet(nn.Module):
  def __init__(self, use_rnn=True):
    super(EyeNet, self).__init__()
    self.resnet = models.resnet18(
        num_classes=128,        # Number of output classes - needs to be defined based on number of eye features needed
        norm_layer=nn.InstanceNorm2d,  # Normalization layer
    )
    self.use_rnn = use_rnn

    # Optional recurrent component - GRUCell
    if use_rnn:
            self.rnn = nn.GRUCell(input_size=128, hidden_size=128)

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

  def forward(self, img_tensor, x, y, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm, screen_w, screen_h, rnn_output=None):

    features = self.resnet(img_tensor)

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
    #
    gaze_direction = (0.5 * np.pi) * self.fc_gaze(features)
    gaze_direction_vector= convert_angles_to_vector(gaze_direction)
    print("gaze direction vector",gaze_direction_vector)
    origin = calculate_gaze_origin_direction(x,y,torch.tensor([0. ,0. ,gaze_direction_vector[0][2]]), z1=0, z2=0)
    point_of_gaze_mm = calculate_intersection_with_screen(origin,gaze_direction_vector,attitude_rotation_matrix)
    print("PoG_mm",(point_of_gaze_mm+point_of_gaze_mm) /2)
    point_of_gaze_px = mm_to_pixels(point_of_gaze_mm, screen_w, screen_h , device_width_mm , device_height_mm)
    pupil_size =self.fc_pupil(features)
    return gaze_direction, pupil_size , point_of_gaze_px

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

def apply_transformation(T, vec):
    if vec.shape[1] == 2:
        vec = convert_angles_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    h_vec = F.pad(vec, pad=(0, 0, 0, 1), value=1.0)
    if T.size(-2) != 4 or T.size(-1) != 4:
        raise ValueError("Transformation matrix T must be of shape [4, 4]")
    return torch.matmul(T, h_vec)[:, :3, 0]


def apply_rotation(T, vec):
    if vec.shape[1] == 2:
        vec = convert_angles_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    if T.dim() == 2:
        T = T.unsqueeze(0)  # Add a batch dimension if it's missing
    elif T.dim() != 3:
        raise ValueError("T must be a 2D or 3D tensor")
    R = T[:, :3, :3]
    return torch.matmul(R, vec).reshape(-1, 3)

# To calculate point of gaze, gaze origin assumed to be 0,0,0
def calculate_intersection_with_screen(o, direction,attitude_rotation_matrix):

    # Ensure o and direction are 2D tensors [N, 3]
    if o.dim() == 1:
        o = o.unsqueeze(0)  # Add batch dimension if necessary
    if direction.dim() == 1:
        direction = direction.unsqueeze(0)  # Add batch dimension if necessary
    
    rows = 3
    cols = 3

    # Convert the flat list to a 3x3 matrix using list comprehension
    matrix = [attitude_rotation_matrix[i * cols:(i + 1) * cols] for i in range(rows)]
    print(matrix)

    attitude_rotation_matrix=torch.tensor(matrix, dtype=torch.float32)

    
    # Assuming no translation, and the camera is at the origin of the world space
    camera_transformation_matrix = torch.eye(4)
    camera_transformation_matrix[:3, :3] = attitude_rotation_matrix
    inverse_camera_transformation_matrix = torch.inverse(camera_transformation_matrix)

    # De-rotate gaze vector
    inv_rotation = torch.inverse(attitude_rotation_matrix)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    direction = apply_rotation(inverse_camera_transformation_matrix, direction)
    o = apply_transformation(inverse_camera_transformation_matrix, o)
    

    # Assuming o = (0, 0, 0) for simplicity
    # Solve for t when z = 0
    epsilon = 1e-6  # Small value to prevent division by zero
    t = -o[:, 2] / (direction[:, 2] + epsilon)

    # Calculate intersection point in millimeters
    p_x = o[:, 0] + t * direction[:, 0]
    p_y = o[:, 1] + t * direction[:, 1]

    return torch.stack([p_x, p_y], dim=-1)

def mm_to_pixels(intersection_mm, device_width_pix , device_height_pix , device_width_mm , device_height_mm):

    # Calculate pixels per millimeter
    ppmm_x = device_width_pix #/ device_width_mm
    ppmm_y = device_height_pix #/ device_height_mm

    # Convert intersection point from mm to pixels
    intersection_px = intersection_mm * torch.tensor([ppmm_x, ppmm_y])
    return intersection_px

def calculate_gaze_origin_direction(x,y,z_gd, z1=0,z2=0):

    direction_vector = torch.tensor([x,y,z1], dtype=torch.float32)
    print("GAze ortigin x y", x,y)
    print("direction vector",direction_vector)

    # Normalize the vector to get a unit vector
    unit_vector = direction_vector / torch.norm(direction_vector)
    print('unit_vector',unit_vector)

    unit_vector= unit_vector + z_gd
    
    print('origin', unit_vector)

    return unit_vector

def average_point_of_gaze(pog1, pog2):
    """
    Calculate the average of two points of gaze.

    Args:
    - pog1: Tuple or list representing the first point of gaze (x1, y1) in pixels.
    - pog2: Tuple or list representing the second point of gaze (x2, y2) in pixels.

    Returns:
    - Tuple representing the average point of gaze (x_avg, y_avg) in pixels.
    """
    # Convert points of gaze to PyTorch tensors for vectorized operations
    pog1_tensor = torch.tensor(pog1, dtype=torch.float32)
    pog2_tensor = torch.tensor(pog2, dtype=torch.float32)

    # Calculate the average point of gaze
    avg_pog_tensor = (pog1_tensor + pog2_tensor) / 2

    avg_pog = avg_pog_tensor.tolist()
    
    print('Avg Pog',avg_pog)

    return avg_pog, avg_pog_tensor

def generate_heatmap(device_height_pix, device_width_pix, pos, sigma=10):
    """
    Generate a Gaussian heatmap centered at pos (x, y).

    :param image_size: Tuple (width, height) of the output image.
    :param pos: Tuple (x, y) position of the gaze point on the screen.
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: Generated heatmap as a 2D numpy array.
    """
    # Create an empty image
    heatmap = np.zeros((device_height_pix, device_width_pix), dtype=np.float32)

    if pos[0][0] < -device_width_pix or pos[0][0] >= device_width_pix or pos[0][1] < -device_height_pix or pos[0][1] >= device_height_pix:
        # count+=1
        # print("HM count",count)
        return heatmap

    # Ensure the position is integer
    pos = np.round(pos).astype(int)
    print("PoG heatmap",pos[0][1],pos[0][0])

    # Set the pixel at the gaze point to 1
    heatmap[pos[0][1], pos[0][0]] = 1

    # Apply Gaussian filter to create the heatmap
    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma, mode='constant')

    # Normalize heatmap
    heatmap /= np.max(heatmap)
    print("heatmap inside function", heatmap)

    return heatmap



def find_gaze_from_heatmap(heatmap):
    """
    Find the approximate gaze point from a heatmap by locating the maximum intensity pixel.

    :param heatmap: A 2D numpy array representing the heatmap.
    :return: Tuple (x, y) position of the approximate gaze point on the screen.
    """
    # Find the index of the maximum value in the heatmap
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Return as (x, y) for consistency
    return (x, y)

def find_gaze_from_heatmap_tensor(heatmap):
    """
    Find the approximate gaze point from a heatmap tensor by locating the maximum intensity pixel.

    :param heatmap: A 2D PyTorch tensor representing the heatmap.
    :return: Tuple (x, y) position of the approximate gaze point on the screen.
    """
    # Find the index of the maximum value in the heatmap
    max_val = torch.max(heatmap)
    idx = (heatmap == max_val).nonzero(as_tuple=True)
    y, x = idx[0].item(), idx[1].item()
    
    # Return as (x, y) for consistency
    return (x, y)

############### GazeRefineNet ############################# 

class GazeRefineNet(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(GazeRefineNet, self).__init__()
        # Dynamically set input channels based on configuration
        in_channels = 4 if config['load_screen_content'] else 1
        self.do_skip = config['use_skip_connections']
        self.eyeNet_r = EyeNet() #Initialized the eye model
        self.eyeNet_l = EyeNet() #Initialized the eye model
        self.landMark = landMark() #Initialized landmark model

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

    def forward(self,kps,screen_w, screen_h,img_tensor_l, lx, ly, img_tensor_r, rx, ry, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm):

        lm_feat = self.landMark(kps) 
        
        gaze_direction_l, pupil_size_l, point_of_gaze_px_l = self.eyeNet_l( img_tensor_l, lx, ly, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm ,screen_w, screen_h) 
        
        gaze_direction_r, pupil_size_r, point_of_gaze_px_r = self.eyeNet_r( img_tensor_r, rx, ry, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm, screen_w, screen_h)

        
        
        average_gaze_direction = (gaze_direction_l + gaze_direction_r) /2

        average_pog , avg_pog_tensor = average_point_of_gaze(point_of_gaze_px_r, point_of_gaze_px_l)

        initial_heatmap = generate_heatmap(device_height_pix, device_width_pix, average_pog, sigma)
        initial_heatmap = torch.tensor(initial_heatmap, dtype=torch.float32).unsqueeze(0)

        print("initial heatmap from forward", initial_heatmap)

        print("Initial heatmap",initial_heatmap)

        # Pass through initial convolutions
        x = self.initial_conv(initial_heatmap)
        # Pass through the backbone
        x = self.backbone(x)
        # Generate final heatmap
        final_heatmap = self.final_conv(x)

        final_heatmap_np = final_heatmap.detach().numpy().squeeze()
        
        
        ### Final PoG from GRN
        PoG_GRN= find_gaze_from_heatmap(final_heatmap_np)
        print("PoG_GRN",PoG_GRN)
        return average_gaze_direction

# Need to change configuration accordingly
config = {
    'load_screen_content': False,
    'use_skip_connections': True,
}