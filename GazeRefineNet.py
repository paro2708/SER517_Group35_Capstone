import torchvision.models as models
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch
import os
import json
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import scipy.ndimage
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from openGazeData import openGazeData

#dimensions for iphone5s
screen_size_mm = [123.8 , 53.7]
screen_size_pixels = [320,568]
screen_size_pixels_heatmap = [640,1136]
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
    # self.image_dir = r'C:\\Rushi\\ProDataset\\train\\images\\iPhone 5S\\cropped_eyes'
    self.image_dir = r'E:\ProDataset\train\images\iPhone 5S\cropped_eyes'
    # self.meta_dir = r'C:\\Rushi\\ProDataset\\train\\meta'
    self.meta_dir = r'E:\ProDataset\train\meta'
    
    #Initializing EyeNet and GazeRefineNet - to be defined in forward
    # self.eyeNet_r = EyeNet() #Initialized the eye model
    # self.eyeNet_l = EyeNet() #Initialized the eye model
    
    self.gazeRefineNet = GazeRefineNet() #Initialized landmark model
    
  #Need to be refined after the entire model is done
  def forward(self, screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r, initial_heatmap=None):
        print("forward grn")
        # gaze_direction_l, pupil_size_l, point_of_gaze_px_l = self.eyeNet_l( img_tensor_l, lx, ly) # need to add screen width and height
        
        # gaze_direction_r, pupil_size_r, point_of_gaze_px_r = self.eyeNet_r( img_tensor_r, rx, ry)

        # average_pog = average_point_of_gaze(point_of_gaze_px_r, point_of_gaze_px_l)

        # initial_heatmap = generate_heatmap(screen_size_pixels_heatmap, average_pog, sigma)
        # initial_heatmap = torch.tensor(initial_heatmap, dtype=torch.float32).unsqueeze(0)

        # print("initial heatmap from forward", initial_heatmap)
        
        # grn_final_PoG, grn_final_heatmap = self.gazeRefineNet(initial_heatmap) 
        average_gaze_direction, grn_final_PoG = self.gazeRefineNet(img_tensor_l, lx, ly, img_tensor_r, rx, ry)
        print("return forward grn")
        # return grn_final_PoG
        return average_gaze_direction
    
  #Need to be refined later
  def training_step(self, batch, batch_idx):
        # _, kps, out, screen_w, screen_h, lx1, lx2, ly1, ly2, rx1, rx2, ry1, ry2, dot_px, device, img_tensor_l, img_tensor_r = batch
        _, kps, out, screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r = batch
        grn_out = self.forward(screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l, img_tensor_r)
        # grn_out_tensor = torch.tensor([grn_out], dtype=torch.float32)
        # dot_px_tensor = torch.stack(dot_px).float()
        # grn_out.requires_grad_(True)
        print("grn_out",grn_out)
        # print("grn_out_tensor",grn_out_tensor)
        print("dot_px",dot_px)
        # output = self.model(inputs, target)
        loss = F.mse_loss(grn_out, out)
        # loss=torch.tensor(3.24).float()
        print('train_loss', loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
  
  def train_dataloader(self):
        print("inside train loader")
        train_dataset = openGazeData(self.image_dir, self.meta_dir)
        print("inside train loader", train_dataset)
        print(self.data_path+ "/train/")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True, persistent_workers=True)
        print('Num_train_files', len(train_dataset))
        return train_loader
    
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
        # pretrained=False,        # Not using pre-trained weights
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

  def forward(self, img_tensor, x, y, rnn_output=None):

    print("Inside eyenet forward")

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
    gaze_direction = (0.5 * np.pi) * self.fc_gaze(features)
    print("GD  ", gaze_direction)
    gaze_direction_vector= convert_angles_to_vector(gaze_direction)
    print("GD vector ", gaze_direction_vector)
    #x1,y1,x2,y2 need to be taken from meta data files for each eye - side to be given as input to eyenet
    print(gaze_direction_vector[0][2])
    # origin = calculate_gaze_origin_direction(x1,x2,y1,y2,torch.tensor([0. ,0. ,gaze_direction_vector[0][2]]), z1=0, z2=0)
    origin = calculate_gaze_origin_direction(x,y,torch.tensor([0. ,0. ,gaze_direction_vector[0][2]]), z1=0, z2=0)
    # print('IMP origin', origin )
    point_of_gaze_mm = calculate_intersection_with_screen(origin,gaze_direction_vector)
    print('PoG mm', point_of_gaze_mm)
    #Hard coding device dimenions from meta data
    #screen_pixels from meta
    
    point_of_gaze_px = mm_to_pixels(point_of_gaze_mm,screen_size_mm, screen_size_pixels) # need to get from screen.json
    pupil_size =self.fc_pupil(features)
    print("Gaze Direction shape before linear layer:", gaze_direction.shape)
    print("Pupil Size shape before linear layer:", pupil_size.shape)
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
def calculate_intersection_with_screen(o, direction):

    # Ensure o and direction are 2D tensors [N, 3]
    if o.dim() == 1:
        o = o.unsqueeze(0)  # Add batch dimension if necessary
    if direction.dim() == 1:
        direction = direction.unsqueeze(0)  # Add batch dimension if necessary
        
    # print('IMP o', o )

    rotation = torch.tensor([
    [0.99970895052,-0.017290327698, 0.0168244000524],
    [-0.0110340490937,0.292467236519, 0.956211805344],
    [-0.0214538034052,-0.956119179726,0.292191326618]
    ], dtype=torch.float32)
    
    
    # Assuming no translation, and the camera is at the origin of the world space
    camera_transformation_matrix = torch.eye(4)
    camera_transformation_matrix[:3, :3] = rotation
    inverse_camera_transformation_matrix = torch.inverse(camera_transformation_matrix)
    
    # print('IMP o', o )

    # De-rotate gaze vector
    inv_rotation = torch.inverse(rotation)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    direction = apply_rotation(inverse_camera_transformation_matrix, direction)
    o = apply_transformation(inverse_camera_transformation_matrix, o)
    
    # print('IMP o', o )

    # Assuming o = (0, 0, 0) for simplicity
    # Solve for t when z = 0
    epsilon = 1e-6  # Small value to prevent division by zero
    t = -o[:, 2] / (direction[:, 2] + epsilon)

    #t = -o[:, 2] / direction[:, 2]
    
    #print('IMP -o[:, 2]', -o[:, 2] )

    # Calculate intersection point in millimeters
    p_x = o[:, 0] + t * direction[:, 0]
    p_y = o[:, 1] + t * direction[:, 1]
    
    #print('IMP p_x', torch.stack([p_x, p_y], dim=-1))
    #print('IMP p_y', p_y )

    return torch.stack([p_x, p_y], dim=-1)

def mm_to_pixels(intersection_mm, screen_size_mm, screen_size_pixels):
    # Unpack screen dimensions
    screen_height_mm, screen_width_mm = screen_size_mm
    screen_height_px, screen_width_px = screen_size_pixels

    # Calculate pixels per millimeter
    ppmm_x = screen_width_px #/ screen_width_mm
    ppmm_y = screen_height_px #/ screen_height_mm

    # Convert intersection point from mm to pixels
    intersection_px = intersection_mm * torch.tensor([ppmm_x, ppmm_y])
    return intersection_px

# def calculate_gaze_origin_direction(x1,x2,y1,y2,z_gd, z1=0, z2=0):
def calculate_gaze_origin_direction(x,y,z_gd, z1=0, z2=0):
    # Convert points to tensors
    # x1 = 293
    # x2 = 346
    # y1 = 406
    # y2 = 405
    # point1 = torch.tensor([x1, y1, z1], dtype=torch.float32)
    # point2 = torch.tensor([x2, y2, z2], dtype=torch.float32)
    # print('lx1',lx1)
    print("Inside gaze origin")
    print("z_gd",z_gd)
    # print('POint 1',x1,y1)
    # point1 = torch.tensor([x1, y1, z1], dtype=torch.float32)
    # print('POint 1',x1,y1)
    # point2 = torch.tensor([x2, y2, z2], dtype=torch.float32)
    # print('POint 2',x2,y2)

    # Calculate the vector pointing from point1 to point2
    # direction_vector = point2 - point1
    direction_vector = torch.tensor([x,y,z1], dtype=torch.float32)

    # Normalize the vector to get a unit vector
    unit_vector = direction_vector / torch.norm(direction_vector)
    print('unit_vector',unit_vector)

    unit_vector= unit_vector + z_gd
    
    print('origin', unit_vector)

    return unit_vector

#eyenet= EyeNet()
# eyenet_r= EyeNet()

# eyenet_l= EyeNet()



# Load and preprocess the image
image_dir = r'C:\\Rushi\\ProDataset\\train\\images\\iPhone 5S\\cropped_eyes'
meta_dir = r'C:\\Rushi\\ProDataset\\train\\meta'
img_tensor = None
lx1, lx2, ly1, ly2 = 0, 0, 0, 0
rx1, rx2, ry1, ry2 = 0, 0, 0, 0

output_json_path = r'C:\\Rushi\\ProDataset\\train\\output'

def loop_through_directory(image_dir, meta_dir):
    print("looping")
    results_list = []
    for meta_file in os.listdir(meta_dir):
        if meta_file.endswith('.json'):
            base_filename = meta_file[:-5]
            #print('base_filename',base_filename)
            left_eye_filename = f"{base_filename}_left_eye.jpg"
            right_eye_filename = f"{base_filename}_right_eye.jpg"
            # print(left_eye_filename)
            left_eye_path = os.path.join(image_dir, left_eye_filename)
            right_eye_path = os.path.join(image_dir, right_eye_filename)
            meta_path = os.path.join(meta_dir, meta_file)
            if os.path.exists(left_eye_path):
                results = process_image_with_metadata(left_eye_path, meta_path)
                results_list.append(results)
            if os.path.exists(right_eye_path):
                results = process_image_with_metadata(right_eye_path, meta_path)
                results_list.append(results)
    with open(output_json_path, 'w') as json_file:
        json.dump(results_list, json_file, indent=4)

def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("preprocessed ")
    return preprocess(img)

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

    # Convert the average point of gaze back to a tuple
    #avg_pog = tuple(avg_pog_tensor.tolist())
    avg_pog = avg_pog_tensor.tolist()
    
    print('Avg Pog',avg_pog)

    return avg_pog

def generate_heatmap(image_size, pos, sigma=10):
    print("Inside generate heatmap")
    print("PoG heatmap",pos)
    """
    Generate a Gaussian heatmap centered at pos (x, y).

    :param image_size: Tuple (width, height) of the output image.
    :param pos: Tuple (x, y) position of the gaze point on the screen.
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: Generated heatmap as a 2D numpy array.
    """
    # Create an empty image
    heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    # Ensure the position is integer
    #pos = [int(p) for p in pos]
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

############### GazeRefineNet


class GazeRefineNet(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(GazeRefineNet, self).__init__()
        # Dynamically set input channels based on configuration
        in_channels = 4 if config['load_screen_content'] else 1
        self.do_skip = config['use_skip_connections']
        self.eyeNet_r = EyeNet() #Initialized the eye model
        self.eyeNet_l = EyeNet() #Initialized the eye model

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

    def forward(self,img_tensor_l, lx, ly, img_tensor_r, rx, ry):
        print("Inside GRN forward")
        # print("Avg pog", average_pog)
        # Example preprocessing step
        # Assuming input_dict contains 'screen_frame' and 'heatmap_initial'

        gaze_direction_l, pupil_size_l, point_of_gaze_px_l = self.eyeNet_l( img_tensor_l, lx, ly) # need to add screen width and height
        
        gaze_direction_r, pupil_size_r, point_of_gaze_px_r = self.eyeNet_r( img_tensor_r, rx, ry)

        average_gaze_direction = (gaze_direction_l + gaze_direction_r) /2


        average_pog = average_point_of_gaze(point_of_gaze_px_r, point_of_gaze_px_l)

        initial_heatmap = generate_heatmap(screen_size_pixels_heatmap, average_pog, sigma)
        initial_heatmap = torch.tensor(initial_heatmap, dtype=torch.float32).unsqueeze(0)

        print("initial heatmap from forward", initial_heatmap)

        print("Initial heatmap",initial_heatmap)

        input_image = initial_heatmap
        # if config['load_screen_content']:
        #     input_image = None
        #     #input_image = torch.cat([input_dict['screen_frame'], input_dict['heatmap_initial']], dim=1)
        # else:
        #     heatmap1 = generate_heatmap(image_size, pos, sigma)
        #     input_image = heatmap1

        # Pass through initial convolutions
        x = self.initial_conv(input_image)
        # Pass through the backbone
        x = self.backbone(x)
        # Generate final heatmap
        final_heatmap = self.final_conv(x)

        final_heatmap_np = final_heatmap.detach().numpy().squeeze()
        
        
        ### Final PoG from GRN
        PoG_GRN= find_gaze_from_heatmap(final_heatmap_np)
        # PoG_GRN= find_gaze_from_heatmap_tensor(final_heatmap)

        

        return average_gaze_direction, PoG_GRN

# Need to change configuration accordingly
config = {
    'load_screen_content': False,
    'use_skip_connections': True,
}
# heatmap1 = generate_heatmap(image_size, pos, sigma)
# heatmap1= torch.from_numpy(heatmap1).float()
# heatmap1= heatmap1.unsqueeze(0).unsqueeze(0)

# Initialize the model
# model = GazeRefineNet(config)
# final_heatmap = model(heatmap1)
# final_heatmap_np = final_heatmap.detach().cpu().numpy().squeeze()
# plt.imshow(final_heatmap_np, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()

#To call eyeNet for left and right eye



#BCE Loss
# final_heatmap_resized = F.interpolate(final_heatmap, size=heatmap1.shape[2:], mode='bilinear', align_corners=False)


# bce_loss = nn.BCELoss()
# m = nn.Sigmoid()
# output = bce_loss(m(final_heatmap_resized), heatmap1)
# output.backward()
# print(output)

def process_image_with_metadata(image_path, meta_path):
    results = {}
    global img_tensor
    img = Image.open(image_path)
    img_tensor = preprocess_image(img)
    with open(meta_path, 'r') as meta_file:
        metadata = json.load(meta_file)
        # lx1 = metadata['leye_x1']
        # ly1 = metadata['leye_y1']
        # lx2 = metadata['leye_x2']
        # ly2 = metadata['leye_y2']
        # rx1 = metadata['reye_x1']
        # ry1 = metadata['reye_y1']
        # rx2 = metadata['reye_x2']
        # ry2 = metadata['reye_y2']
        # dot_px = [metadata['dot_x_pix'],metadata['dot_y_pix']]
        #gaze_direction, pupil_size, point_of_gaze_px = eyenet(img_tensor.unsqueeze(0),lx1,lx2)
        gaze_direction_r, pupil_size_r, point_of_gaze_px_r = eyenet_r(img_tensor.unsqueeze(0),rx1, rx2, ry1, ry2)
        gaze_direction_l, pupil_size_l, point_of_gaze_px_l = eyenet_l( img_tensor.unsqueeze(0),lx1, lx2, ly1, ly2)
        print("Predicted Point of Gaze R:", point_of_gaze_px_r)
        print("Predicted Point of Gaze L:", point_of_gaze_px_l)
        
        # Calculate the average point of gaze
        average_pog = average_point_of_gaze(point_of_gaze_px_r, point_of_gaze_px_l)
        heatmap_initial = generate_heatmap(screen_size_pixels_heatmap, average_pog, sigma )
        ## save heatmap to results
        heatmap_initial= torch.from_numpy(heatmap_initial).float()
        heatmap_initial= heatmap_initial.unsqueeze(0).unsqueeze(0)
        
        ### GazeREfinenet training and initialisation
        GRNmodel = GazeRefineNet(config)
        final_heatmap = GRNmodel(heatmap_initial)
        # final_heatmap_np = final_heatmap.detach().cpu().numpy().squeeze()
        final_heatmap_np = final_heatmap.detach().numpy().squeeze()
        
        ### Final PoG from GRN
        PoG_GRN= find_gaze_from_heatmap(final_heatmap_np)

        # print("Predicted Gaze Direction:", gaze_direction_r)
        # print("Predicted Gaze Direction:", gaze_direction_l)
        # print("Predicted Pupil Size:", pupil_size)
        # print("Predicted Point of Gaze R:", point_of_gaze_px_r)
        # print("Predicted Point of Gaze L:", point_of_gaze_px_l)
        # magnitude = torch.norm(gaze_direction, p=2)
        # magnitude = magnitude *(180/np.pi)
        # print("Normalized Gaze Direction Magnitude(in radians):", magnitude.item())
        print('Average PoG',average_pog)
        print('Final PoG',PoG_GRN)

        results['image_path'] = image_path
        results['gaze_direction_l'] = gaze_direction_l.tolist()
        results['gaze_direction_r'] = gaze_direction_r.tolist()
        results['pupil_size_l'] = pupil_size_l.item()
        results['pupil_size_r'] = pupil_size_r.item()
        results['average_pog'] = average_pog.tolist()
        results['final_pog'] = PoG_GRN.item()
    
    return results
    # print("lx", lx1)
    # print(f"Processed {image_path} using {meta_path}")

# loop_through_directory(image_dir, meta_dir)

# gaze_direction, pupil_size, point_of_gaze_px = eyenet(img_tensor.unsqueeze(0))

# print("Predicted Gaze Direction:", gaze_direction)
# print("Predicted Pupil Size:", pupil_size)
# print("Predicted Point of Gaze:", point_of_gaze_px)
# magnitude = torch.norm(gaze_direction, p=2)
# magnitude = magnitude *(180/np.pi)
# print("Normalized Gaze Direction Magnitude(in radians):", magnitude.item())
