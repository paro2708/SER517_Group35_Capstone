from torch.utils.data import Dataset
from glob import glob
import json
from PIL import Image
import torch
from torchvision.transforms import Normalize, Resize, Compose, ToTensor, RandomCrop
import matplotlib.pyplot as plt



class openGazeData(Dataset):
    def __init__(self, root, phase='train', size = (128,128), transform=True):
        self.root = root
        print("Root = ", root)
        self.files = glob(root+"/images/*.jpg")

        self.phase = phase
        self.size = size

        self.aug = self.get_transform(self.phase, self.size)
        self.transform = transform

        print("Num Files for " + phase + " = " + str(len(self.files)))
    
    def __getitem__(self,idx):
        image = Image.open(self.files[idx])
        fname = self.files[idx]
        with open(self.files[idx].replace('.jpg','.json').replace('images', 'meta')) as f:
            meta = json.load(f)
        w, h = image.size
        screen_w, screen_h = meta['screen_w'], meta['screen_h']
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        
        kps = [meta['leye_x1']/w, meta['leye_y1']/h, meta['leye_x2']/w, meta['leye_y2']/h, 
               meta['reye_x1']/w, meta['reye_y1']/h, meta['reye_x2']/w, meta['reye_y2']/h]
        
        l_eye = image.crop((max(0, lx), max(0, ly), max(0, lx+lw), max(0, ly+lh)))
        r_eye = image.crop((max(0, rx), max(0, ry), max(0, rx+rw), max(0, ry+rh)))
        
        l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)
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
        l_eye_w, l_eye_h = l_eye.size
        r_eye_w, r_eye_h = r_eye.size

        # Extract left and right eye keypoint pixel coordinates
        l_kps = [kps[0].item() * l_eye_w, kps[1].item() * l_eye_h, kps[2].item() * l_eye_w, kps[3].item() * l_eye_h]
        r_kps = [kps[4].item() * r_eye_w, kps[5].item() * r_eye_h, kps[6].item() * r_eye_w, kps[7].item() * r_eye_h]

        # Display images with keypoints
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Left eye and keypoints
        ax[0].imshow(l_eye)
        ax[0].scatter([l_kps[0], l_kps[2]], [l_kps[1], l_kps[3]], color='red')  # Assuming l_kps[0] and l_kps[2] are x coordinates, l_kps[1] and l_kps[3] are y
        ax[0].set_title('Left Eye with Keypoints')
        ax[0].axis('off')

        # Right eye and keypoints
        ax[1].imshow(r_eye)
        ax[1].scatter([r_kps[0], r_kps[2]], [r_kps[1], r_kps[3]], color='red')  # Same assumption for r_kps
        ax[1].set_title('Right Eye with Keypoints')
        ax[1].axis('off')

        plt.show()
        out = torch.tensor([meta['dot_xcam'], meta['dot_y_cam']]).float()
        
        l_eye = self.aug(l_eye)
        r_eye = self.aug(r_eye)
        
        return self.files[idx], l_eye, r_eye, kps, out, screen_w, screen_h
    
    def get_transform(self,phase, size):
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
    
    def __len__(self):
        return len(self.files)