import cv2
import os
import json
import dlib
from glob import glob 
from multiprocessing import Pool, Process
from tqdm import tqdm

'''
Add eye landmarks to the MIT Gaze Capture Dataset using DLib
'''

device_dimensions = {
    'iPhone 6': {'width_pix': 750, 'height_pix': 1334, 'width_mm': 67.0, 'height_mm': 138.1},
    'iPhone 6s Plus': {'width_pix': 1080, 'height_pix': 1920, 'width_mm': 77.9, 'height_mm': 158.2},
    'iPhone 5S': {'width_pix': 640, 'height_pix': 1136, 'width_mm': 58.6, 'height_mm': 123.8},
    'iPad Mini': {'width_pix': 768, 'height_pix': 1024, 'width_mm': 134.8, 'height_mm': 200.0}, # Assuming iPad Mini 1st gen
    'iPhone 4S': {'width_pix': 640, 'height_pix': 960, 'width_mm': 58.6, 'height_mm': 115.2},
    'iPhone 5C': {'width_pix': 640, 'height_pix': 1136, 'width_mm': 59.2, 'height_mm': 124.4},
    'iPad Air 2': {'width_pix': 1536, 'height_pix': 2048, 'width_mm': 169.5, 'height_mm': 240.0},
    'iPhone 6s': {'width_pix': 750, 'height_pix': 1334, 'width_mm': 67.1, 'height_mm': 138.3},
    'iPad Pro 12.9"': {'width_pix': 2048, 'height_pix': 2732, 'width_mm': 220.6, 'height_mm': 305.7},
    'iPhone 6 Plus': {'width_pix': 1080, 'height_pix': 1920, 'width_mm': 77.8, 'height_mm': 158.1},
    'iPad 4': {'width_pix': 1536, 'height_pix': 2048, 'width_mm': 185.7, 'height_mm': 241.2},
    'iPhone 5': {'width_pix': 640, 'height_pix': 1136, 'width_mm': 58.6, 'height_mm': 123.8},
    'iPad Air': {'width_pix': 1536, 'height_pix': 2048, 'width_mm': 169.5, 'height_mm': 240.0},
    'iPad 2': {'width_pix': 768, 'height_pix': 1024, 'width_mm': 185.7, 'height_mm': 241.2},
}


def in_box(box, point):
    x1, y1, w, h = box
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

def add_kps(files, p):
    # Initialize DLib's face detector and shape predictor with the given model.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    no_face = 0  # Number of images where no face was detected.
    err_ctr = 0  # Number of images where landmarks were not correctly identified within the eye boxes.
    buffer = 10  # Buffer to increase the eye bounding box to ensure landmark detection within this area.
    for i in tqdm(files):
        img = cv2.imread(i)
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meta = json.load(open(i.replace('images', 'meta').replace('.jpg', '.json')))

        meta_file_path = i.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file_path, 'r') as meta_file:
            meta = json.load(meta_file)
        
        device = meta['device']
        # Update metadata with device dimensions if the device is known
        if device in device_dimensions:
            meta.update({
                'device_width_pix': device_dimensions[device]['width_pix'],
                'device_height_pix': device_dimensions[device]['height_pix'],
                'device_width_mm': device_dimensions[device]['width_mm'],
                'device_height_mm': device_dimensions[device]['height_mm']
            })
        else:
            # Handle unknown device case, maybe log it or use default dimensions
            print(f"Unknown device: {device}")
            
        leye_x, leye_y, leye_w, leye_h = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        reye_x, reye_y, reye_w, reye_h = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        if(meta['face_valid']):
            fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
        else:
            no_face += 1
            faces = detector(bw_img, 1)
            for face in faces:
                if(face.left()<reye_x and face.top()<reye_y and face.right()>leye_x and face.bottom()>leye_y+leye_h):
                    fx, fy, fw, fh = face.left(), face.top(), face.right()-face.left, face.bottom()-face.top()
                    break
        # Detect landmarks within the detected face region.
        face_rect = dlib.rectangle(fx, fy, fx+fw, fy+fh)
        kps = predictor(bw_img, face_rect)
        
        # make sure the detected landmark point are within the eye bounding box.
        if(in_box((reye_x-buffer, reye_y-buffer, reye_w+(buffer*2), reye_h+(buffer*2)), (kps.part(36).x, kps.part(36).y)) 
           and in_box((reye_x-buffer, reye_y-buffer, reye_w+(buffer*2), reye_h*(buffer*2)), (kps.part(39).x, kps.part(39).y))):
            
            meta['reye_x1'], meta['reye_y1'], meta['reye_x2'], meta['reye_y2'] = kps.part(36).x, kps.part(36).y, kps.part(39).x, kps.part(39).y
        else:
            err_ctr+=1
            meta['reye_x1'], meta['reye_y1'], meta['reye_x2'], meta['reye_y2'] = reye_x, reye_y+(reye_h//2), reye_x+reye_w, reye_y+(reye_h//2)
            
        if(in_box((leye_x-buffer, leye_y-buffer, leye_w+(buffer*2), leye_h+(buffer*2)), (kps.part(42).x, kps.part(42).y)) 
           and in_box((leye_x-buffer, leye_y-buffer, leye_w+(buffer*2), leye_h+(buffer*2)), (kps.part(45).x, kps.part(45).y))):
            meta['leye_x1'], meta['leye_y1'], meta['leye_x2'], meta['leye_y2'] = kps.part(42).x, kps.part(42).y, kps.part(45).x, kps.part(45).y
        else:
            err_ctr+=1
            meta['leye_x1'], meta['leye_y1'], meta['leye_x2'], meta['leye_y2'] = leye_x, leye_y+(leye_h//2), leye_x+leye_w, leye_y+(leye_h//2)
    
        meta_file = i.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file, 'w') as outfile:
            json.dump(meta, outfile)
    print("face errs: ", no_face)
    
def assign_work(path, p):
    files = glob(path+"**/images/*.jpg")
    add_kps(files,p)

def main():
    data_dir = "../ProDataset/"
    model = "../shape_predictor_68_face_landmarks.dat"
    assign_work(data_dir, model)
    print("KPs Added")
    
if __name__ == '__main__':
    main()

