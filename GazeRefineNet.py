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


class GRN(pl.LightningModule):
  def __init__(self, data_path, save_path):
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
    # self.gazeRefineNet = gazeRefineNet() #Initialized landmark model
    
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
  def __init__(self, side='left', use_rnn=True):
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
    #x1,y1,x2,y2 need to be taken from meta data files for each eye - side to be given as input to eyenet
    print(gaze_direction_vector[0][2])
    origin = calculate_gaze_origin_direction(torch.tensor([0. ,0. ,gaze_direction_vector[0][2]]), z1=0, z2=0)
    point_of_gaze_mm = calculate_intersection_with_screen(origin,gaze_direction_vector)
    #Hard coding device dimenions from meta data
    #screen_pixels from meta
    screen_size_mm = [123.8 , 53.7]
    screen_size_pixels = [568 , 320]
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

    rotation = torch.tensor([
    [0.99970895052,-0.017290327698, 0.0168244000524],
    [-0.0110340490937,0.292467236519, 0.9562118053443],
    [-0.0214538034052,-0.956119179726,0.292191326618]
    ], dtype=torch.float32)

    # Assuming no translation, and the camera is at the origin of the world space
    camera_transformation_matrix = torch.eye(4)
    camera_transformation_matrix[:3, :3] = rotation
    inverse_camera_transformation_matrix = torch.inverse(camera_transformation_matrix)

    # De-rotate gaze vector
    inv_rotation = torch.inverse(rotation)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    direction = apply_rotation(inverse_camera_transformation_matrix, direction)
    o = apply_transformation(inverse_camera_transformation_matrix, o)

    # Assuming o = (0, 0, 0) for simplicity
    # Solve for t when z = 0
    t = -o[:, 2] / direction[:, 2]

    # Calculate intersection point in millimeters
    p_x = o[:, 0] + t * direction[:, 0]
    p_y = o[:, 1] + t * direction[:, 1]

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

def calculate_gaze_origin_direction(z_gd, z1=0, z2=0):
    # Convert points to tensors
    # x1 = 293
    # x2 = 346
    # y1 = 406
    # y2 = 405
    # point1 = torch.tensor([x1, y1, z1], dtype=torch.float32)
    # point2 = torch.tensor([x2, y2, z2], dtype=torch.float32)
    point1 = torch.tensor([lx1, ly1, z1], dtype=torch.float32)
    point2 = torch.tensor([lx2, ly2, z2], dtype=torch.float32)

    # Calculate the vector pointing from point1 to point2
    direction_vector = point2 - point1

    # Normalize the vector to get a unit vector
    unit_vector = direction_vector / torch.norm(direction_vector)

    unit_vector= unit_vector + z_gd

    return unit_vector

eyenet= EyeNet()

# Load and preprocess the image
image_dir = r'C:\\Users\\Paromita Roy\\OneDrive\\Documents\\Coursework\\Capstone\\SER517_Group35_Capstone\\ProDataset\\train\\images\\cropped_eyes'
meta_dir = r'C:\\Users\\Paromita Roy\\OneDrive\\Documents\\Coursework\\Capstone\\SER517_Group35_Capstone\\ProDataset\\train\\meta'
img_tensor = None
lx1, lx2, ly1, ly2 = 0, 0, 0, 0
rx1, rx2, ry1, ry2 = 0, 0, 0, 0

output_json_path = r'C:\\Users\\Paromita Roy\\OneDrive\\Documents\\Coursework\\Capstone\\SER517_Group35_Capstone\\ProDataset\\train\\output'

def loop_through_directory(image_dir, meta_dir, output_json_path):
    results_list = []
    print("looping")
    for meta_file in os.listdir(meta_dir):
        if meta_file.endswith('.json'):
            base_filename = meta_file[:-5]
            left_eye_filename = f"{base_filename}_left_eye.jpg"
            right_eye_filename = f"{base_filename}_right_eye.jpg"
            left_eye_path = os.path.join(image_dir, left_eye_filename)
            right_eye_path = os.path.join(image_dir, right_eye_filename)
            meta_path = os.path.join(meta_dir, meta_file)
            if os.path.exists(left_eye_path):
                results = process_image_with_metadata(left_eye_path, meta_path)
                results_list.append(results)
            if os.path.exists(right_eye_path):
                results = process_image_with_metadata(right_eye_path, meta_path)
                results_list.append(results)
            # if os.path.exists(left_eye_path):
            #     process_image_with_metadata(left_eye_path, meta_path)
            #     # print("left eye", left_eye_path)
            # if os.path.exists(right_eye_path):
            #     process_image_with_metadata(right_eye_path, meta_path)
            #     # print("right eye", right_eye_path)
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

def process_image_with_metadata(image_path, meta_path):
    results = {}
    global img_tensor
    img = Image.open(image_path)
    img_tensor = preprocess_image(img)
    with open(meta_path, 'r') as meta_file:
        metadata = json.load(meta_file)
        lx1 = metadata['leye_x1']
        ly1 = metadata['leye_y1']
        lx2 = metadata['leye_x2']
        ly2 = metadata['leye_y2']
        rx1 = metadata['reye_x1']
        ry1 = metadata['reye_y1']
        rx2 = metadata['reye_x2']
        ry2 = metadata['reye_y2']
        gaze_direction, pupil_size, point_of_gaze_px = eyenet(img_tensor.unsqueeze(0))

        print("Predicted Gaze Direction:", gaze_direction)
        print("Predicted Pupil Size:", pupil_size)
        print("Predicted Point of Gaze:", point_of_gaze_px)
        magnitude = torch.norm(gaze_direction, p=2)
        magnitude = magnitude *(180/np.pi)
        print("Normalized Gaze Direction Magnitude(in radians):", magnitude.item())

        results['image_path'] = image_path
        results['gaze_direction'] = gaze_direction.tolist()
        results['pupil_size'] = pupil_size.item()
        results['point_of_gaze'] = point_of_gaze_px.tolist()
        results['magnitude'] = magnitude.item()
    
    return results
    # print("lx", lx1)
    # print(f"Processed {image_path} using {meta_path}")

loop_through_directory(image_dir, meta_dir, output_json_path)

# gaze_direction, pupil_size, point_of_gaze_px = eyenet(img_tensor.unsqueeze(0))

# print("Predicted Gaze Direction:", gaze_direction)
# print("Predicted Pupil Size:", pupil_size)
# print("Predicted Point of Gaze:", point_of_gaze_px)
# magnitude = torch.norm(gaze_direction, p=2)
# magnitude = magnitude *(180/np.pi)
# print("Normalized Gaze Direction Magnitude(in radians):", magnitude.item())