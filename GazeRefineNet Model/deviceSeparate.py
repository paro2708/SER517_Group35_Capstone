import os
import json
import shutil

# Define the paths to your metadata and images directories
base_dir = r'C:\\Users\\Paromita Roy\\OneDrive\\Documents\\Coursework\\Capstone\\SER517_Group35_Capstone\\ProDataset\\train'
meta_dir = os.path.join(base_dir, 'meta')
images_dir = os.path.join(base_dir, 'images')

# Loop through each file in the metadata directory
for filename in os.listdir(meta_dir):
    if filename.endswith('.json'):  # Process only JSON files
        json_path = os.path.join(meta_dir, filename)
        
        # Open and read the JSON file
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            device = data['device']  # Extract the device value
            
            # Create a folder for the device within the images directory if it doesn't exist
            device_folder = os.path.join(images_dir, device)
            if not os.path.exists(device_folder):
                os.makedirs(device_folder)
            
            # Construct the image file name from the JSON file name
            image_filename = filename.replace('.json', '.jpg')
            source_image_path = os.path.join(images_dir, image_filename)
            destination_image_path = os.path.join(device_folder, image_filename)
            
            # Move the image file to the device-specific folder
            if os.path.exists(source_image_path):  # Check if the image file exists
                shutil.move(source_image_path, destination_image_path)
