import glob
import os
import cv2
import json

train_images_path = "C:\\Rushi\\ProDataset\\train\\images\\iPhone 5S"
train_metadata_path = "C:\\Rushi\\ProDataset\\train\\meta"

train_output_path = "C:\\Rushi\\ProDataset\\train\\images\\iPhone 5S\\cropped_eyes"

if not os.path.exists(train_output_path):
    os.makedirs(train_output_path)
    
for image_filename in glob.glob(os.path.join(train_images_path, "*.jpg")):
    base_filename = os.path.basename(image_filename)
    metadata_filename = os.path.join(train_metadata_path, base_filename.replace(".jpg", ".json"))
    
    if not os.path.exists(metadata_filename):
        print(f"Metadata file does not exist for image {image_filename}")
        continue
    
    print("Continuing coz metadata exists")
    
    img = cv2.imread(image_filename)
    print(img)

    with open(metadata_filename, 'r') as f:
        metadata = json.load(f)
    
    leye_roi = img[metadata['leye_y']:metadata['leye_y']+metadata['leye_h'], metadata['leye_x']:metadata['leye_x']+metadata['leye_w']]
    reye_roi = img[metadata['reye_y']:metadata['reye_y']+metadata['reye_h'], metadata['reye_x']:metadata['reye_x']+metadata['reye_w']]

    leye_resized = cv2.resize(leye_roi, (128, 128))
    reye_resized = cv2.resize(reye_roi, (128, 128))

    basename = os.path.basename(image_filename).split('.')[0]
    cv2.imwrite(os.path.join(train_output_path, f"{basename}_left_eye.jpg"), leye_resized)
    cv2.imwrite(os.path.join(train_output_path, f"{basename}_right_eye.jpg"), reye_resized)