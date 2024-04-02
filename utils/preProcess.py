import json
import os
import numpy as np
from glob import glob

'''
Frames from MIT Dataset, split in train/test.
Create a metadata for each image 
'''

# device_dimensions = {
#     'iPhone12': {'width': 750, 'height': 1334},
#     'GalaxyS21': {'width': 1080, 'height': 2400},
    
#     # Add other devices
# }

def convert_dataset(files,out_root):
    devices_set = set()
    for i in files:
        with open(i+"/info.json") as f:
            data = json.load(f)
            ds = data['Dataset']
            device = data['DeviceName']
        devices_set.add(device)
        out_dir = out_root+ds
        expt_name = i.split('\\')[-2]
        screen_info = json.load(open(i+'/screen.json'))
        face_det = json.load(open(i+'/appleFace.json'))
        l_eye_det = json.load(open(i+'/appleLeftEye.json'))
        r_eye_det = json.load(open(i+'/appleRightEye.json'))
        dot = json.load(open(i+'/dotInfo.json'))
        motion = json.load(open(i+'/motion.json'))
        
        # Determine frames where the device is in portrait orientation and both eyes are detected.

        attitude_rotation_matrix = motion[0]['AttitudeRotationMatrix'] if 'AttitudeRotationMatrix' in motion[0] else None
        portrait_orientation = np.asarray(screen_info["Orientation"])==1
        l_eye_valid, r_eye_valid = np.array(l_eye_det['IsValid']), np.array(r_eye_det['IsValid'])
        valid_ids = l_eye_valid*r_eye_valid*portrait_orientation
        
        frame_ids = np.where(valid_ids==1)[0]
        for frame_idx in frame_ids:
            fname = str(frame_idx).zfill(5)
            src,target = i+'/frames/'+fname+".jpg", out_dir+"/images/"+expt_name+'__'+fname+'.jpg'
            copy_file_manual(src,target)
           
            # Prepare metadata for the current frame. 
            meta = {}
            meta['device'] = device
            # Look up device dimensions based on the device name
            # if device in device_dimensions:
            #     meta['device_w'] = device_dimensions[device]['width']
            #     meta['device_h'] = device_dimensions[device]['height']
            # else:
                # Default dimensions or a method to handle unknown devices
            #     meta['device_w'] = 0 
            #     meta['device_h'] = 0 
            meta['screen_h'], meta['screen_w'] = screen_info["H"][frame_idx], screen_info["W"][frame_idx]
            meta['face_valid'] = face_det["IsValid"][frame_idx]
            meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h'] = round(face_det['X'][frame_idx]), round(face_det['Y'][frame_idx]), round(face_det['W'][frame_idx]), round(face_det['H'][frame_idx])
            meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h'] = meta['face_x']+round(l_eye_det['X'][frame_idx]), meta['face_y']+round(l_eye_det['Y'][frame_idx]), round(l_eye_det['W'][frame_idx]), round(l_eye_det['H'][frame_idx])
            meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h'] = meta['face_x']+round(r_eye_det['X'][frame_idx]), meta['face_y']+round(r_eye_det['Y'][frame_idx]), round(r_eye_det['W'][frame_idx]), round(r_eye_det['H'][frame_idx])
            
            meta['dot_xcam'], meta['dot_y_cam'] = dot['XCam'][frame_idx], dot['YCam'][frame_idx]
            meta['dot_x_pix'], meta['dot_y_pix'] = dot['XPts'][frame_idx], dot['YPts'][frame_idx]
            
            # Add the first AttitudeRotationMatrix to the metadata
            if attitude_rotation_matrix:
                meta['attitude_rotation_matrix'] = attitude_rotation_matrix

            os.makedirs(out_dir+'/meta/', exist_ok=True)
            meta_file = out_dir+'/meta/'+expt_name+'__'+fname+'.json'
            with open(meta_file, 'w') as outfile:
                json.dump(meta, outfile)
        print(i + " completed. Images = " + str(len(frame_ids)))
    devices_list = list(devices_set)
    print("devices_list ", devices_list)

    with open('unique_devices.txt', 'w') as txt_file:
        for device in devices_list:
            txt_file.write(device + "\n")
    print("Unique device names written to unique_devices.txt")
    return 0


def assign_work(path,out_dir):
    files = glob(path+"/*/")
    convert_dataset(files,out_dir)

def copy_file_manual(src_path, dest_path):
    # Ensure the target directory exists
    dest_dir = os.path.dirname(dest_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    # Open the source file and read its content, then write it to the destination file
    with open(src_path, 'rb+') as src_file:
        content = src_file.read()
        with open(dest_path, 'wb+') as dest_file:
            dest_file.write(content)

def main():
    dataset_dir = "../Dataset/"
    out_dir = "../ProDataset/"
    print(os.path.exists(dataset_dir))
    assign_work(dataset_dir, out_dir)
    print("called")

if __name__ == "__main__":
    main()