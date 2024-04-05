from torch.utils.data import Dataset
from glob import glob
import json
from PIL import Image
import torch
from torchvision.transforms import Normalize, Resize, Compose, ToTensor, RandomCrop, CenterCrop
import matplotlib.pyplot as plt
import os


class openGazeData(Dataset):
    def __init__(self, image_dir, meta_dir):
        self.image_dir =  image_dir
        self.meta_dir = meta_dir
        self.data = []
        self.load_data()
        print("init finished")
    
    def __getitem__(self,idx):
        print("preprocessed")
        left_eye_path, right_eye_path, meta = self.data[idx]
        
        left_eye_img = Image.open(left_eye_path)
        right_eye_img = Image.open(right_eye_path)
        
        # left_eye_img = self.preprocess(left_eye_img)
        # right_eye_img = self.preprocess(right_eye_img)
        # with open(self.files[idx].replace('.jpg','.json').replace('images', 'meta')) as f:
        #     meta = json.load(f)
        
        
        w, h = left_eye_img.size
        screen_w, screen_h = meta['screen_w'], meta['screen_h']
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']

        lx1 = meta['leye_x1']
        ly1 = meta['leye_y1']
        lx2 = meta['leye_x2']
        ly2 = meta['leye_y2']
        rx1 = meta['reye_x1']
        ry1 = meta['reye_y1']
        rx2 = meta['reye_x2']
        ry2 = meta['reye_y2']
        dot_px = torch.tensor([meta['dot_x_pix'],meta['dot_y_pix']]).float()
        attitude_rotation_matrix = meta['attitude_rotation_matrix']
        device_width_pix = meta['device_width_pix']
        device_height_pix = meta['device_height_pix']
        device_width_mm = meta['device_width_mm']
        device_height_mm = meta['device_height_mm']
        print("Dot_px",dot_px)
        device = meta['device']
        img_tensor_l = self.preprocess_image(left_eye_img)
        img_tensor_l = img_tensor_l#.unsqueeze(0)

        img_tensor_r = self.preprocess_image(right_eye_img)
        img_tensor_r = img_tensor_r#.unsqueeze(0)

        kps = [meta['leye_x1']/w, meta['leye_y1']/h, meta['leye_x2']/w, meta['leye_y2']/h, 
               meta['reye_x1']/w, meta['reye_y1']/h, meta['reye_x2']/w, meta['reye_y2']/h]
        
        # l_eye = image.crop((max(0, lx), max(0, ly), max(0, lx+lw), max(0, ly+lh)))
        # r_eye = image.crop((max(0, rx), max(0, ry), max(0, rx+rw), max(0, ry+rh)))
        
        # l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)
        # plt.figure(figsize=(2, 2))  # Size is arbitrary; adjust for your needs
        # plt.imshow(l_eye)
        # plt.title('Left Eye')
        # plt.axis('off')  # Turn off axis numbers and ticks
        # plt.show()

        # # To display the right eye image
        # plt.figure(figsize=(2, 2))  # Size is arbitrary; adjust for your needs
        # plt.imshow(r_eye)
        # plt.title('Right Eye')
        # plt.axis('off')  # Turn off axis numbers and ticks
        # plt.show()
        kps = torch.tensor(kps).float()
        # l_eye_w, l_eye_h = l_eye.size
        # r_eye_w, r_eye_h = r_eye.size

        # # Extract left and right eye keypoint pixel coordinates
        # l_kps = [kps[0].item() * l_eye_w, kps[1].item() * l_eye_h, kps[2].item() * l_eye_w, kps[3].item() * l_eye_h]
        # r_kps = [kps[4].item() * r_eye_w, kps[5].item() * r_eye_h, kps[6].item() * r_eye_w, kps[7].item() * r_eye_h]

        # # Display images with keypoints
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # # Left eye and keypoints
        # ax[0].imshow(l_eye)
        # ax[0].scatter([l_kps[0], l_kps[2]], [l_kps[1], l_kps[3]], color='red')  # Assuming l_kps[0] and l_kps[2] are x coordinates, l_kps[1] and l_kps[3] are y
        # ax[0].set_title('Left Eye with Keypoints')
        # ax[0].axis('off')

        # # Right eye and keypoints
        # ax[1].imshow(r_eye)
        # ax[1].scatter([r_kps[0], r_kps[2]], [r_kps[1], r_kps[3]], color='red')  # Same assumption for r_kps
        # ax[1].set_title('Right Eye with Keypoints')
        # ax[1].axis('off')

        # plt.show()
        out = torch.tensor([meta['dot_xcam'], meta['dot_y_cam']]).float()
        
        # l_eye = self.aug(l_eye)
        # r_eye = self.aug(r_eye)
        lx=lx2-lx1
        ly = ly2-ly1
        rx = rx2-rx1
        ry=ry2 - ry1
        
        # return self.data[idx], kps, out, screen_w, screen_h, lx1, lx2, ly1, ly2, rx1, rx2, ry1, ry2, dot_px, device, img_tensor_l , img_tensor_r
        return self.data[idx], kps, out, screen_w, screen_h, lx, ly, rx, ry, dot_px, device, img_tensor_l , img_tensor_r, attitude_rotation_matrix, device_width_pix , device_height_pix , device_width_mm , device_height_mm
    
    def get_transform(self,phase, size):
        print("get transform")
        list_transforms = []
        if(phase=="train"):
            list_transforms = [Resize((size[0]+10,size[1]+10)),
                               RandomCrop((size[0],size[1])),
                               ToTensor(),
                               Normalize(mean=(0.3741, 0.4076, 0.5425), std=(0.02, 0.02, 0.02)),]
            
        else:
            list_transforms = [Resize((size[0],size[1])),
                               ToTensor(),
                               Normalize(mean=(0.3741, 0.4076, 0.5425), std=(0.02, 0.02, 0.02)),]
        
        list_trfms = Compose(list_transforms)
        return list_trfms
    
    def preprocess_image(self,img):
        # print("preprocess image")
        preprocess = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # print("preprocessed ")
        return preprocess(img)
    
    def load_data(self):
        # Similar to loop_through_directory but adapted for dataset initialization
        # print("inside load data")
        for meta_file in os.listdir(self.meta_dir):
            if meta_file.endswith('.json'):
                base_filename = meta_file[:-5]
                left_eye_filename = f"{base_filename}_left_eye.jpg"
                right_eye_filename = f"{base_filename}_right_eye.jpg"
                left_eye_path = os.path.join(self.image_dir, left_eye_filename)
                right_eye_path = os.path.join(self.image_dir, right_eye_filename)
                meta_path = os.path.join(self.meta_dir, meta_file)
                print("File",base_filename)

                # print('left eye path',left_eye_path)
                
                if os.path.exists(left_eye_path) and os.path.exists(right_eye_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    self.data.append((left_eye_path, right_eye_path, meta))
                    # print(self.data)


    
    def __len__(self):
        return len(self.data)