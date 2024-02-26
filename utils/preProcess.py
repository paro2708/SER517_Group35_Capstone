import json
import os
import numpy as np
from glob import glob

'''
Frames from MIT Dataset, split in train/test.
Create a metadata for each image 
'''

def convert_dataset(files,out_root):
    for i in files:
        with open(i+"/info.json") as f:
            data = json.load(f)
            ds = data['Dataset']
            device = data['DeviceName']
        out_dir = out_root+ds
        expt_name = i.split('\\')[-2]
        screen_info = json.load(open(i+'/screen.json'))
        face_det = json.load(open(i+'/appleFace.json'))
        l_eye_det = json.load(open(i+'/appleLeftEye.json'))
        r_eye_det = json.load(open(i+'/appleRightEye.json'))
        dot = json.load(open(i+'/dotInfo.json'))
        
        # Determine frames where the device is in portrait orientation and both eyes are detected.

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
            meta['screen_h'], meta['screen_w'] = screen_info["H"][frame_idx], screen_info["W"][frame_idx]
            meta['face_valid'] = face_det["IsValid"][frame_idx]
            meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h'] = round(face_det['X'][frame_idx]), round(face_det['Y'][frame_idx]), round(face_det['W'][frame_idx]), round(face_det['H'][frame_idx])
            meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h'] = meta['face_x']+round(l_eye_det['X'][frame_idx]), meta['face_y']+round(l_eye_det['Y'][frame_idx]), round(l_eye_det['W'][frame_idx]), round(l_eye_det['H'][frame_idx])
            meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h'] = meta['face_x']+round(r_eye_det['X'][frame_idx]), meta['face_y']+round(r_eye_det['Y'][frame_idx]), round(r_eye_det['W'][frame_idx]), round(r_eye_det['H'][frame_idx])
            
            meta['dot_xcam'], meta['dot_y_cam'] = dot['XCam'][frame_idx], dot['YCam'][frame_idx]
            meta['dot_x_pix'], meta['dot_y_pix'] = dot['XPts'][frame_idx], dot['YPts'][frame_idx]
            
            os.makedirs(out_dir+'/meta/', exist_ok=True)
            meta_file = out_dir+'/meta/'+expt_name+'__'+fname+'.json'
            with open(meta_file, 'w') as outfile:
                json.dump(meta, outfile)
        print(i + " completed. Images = " + str(len(frame_ids)))
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