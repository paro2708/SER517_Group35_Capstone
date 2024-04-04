import os
import shutil

# Base directory path where your dataset directories are located
base_directory_path = '/home/ec2-user'

for root, dirs, _ in os.walk(base_directory_path):
    # Check if 'frames' directory exists in the current root directory
    if 'frames' in dirs:
        frames_dir_path = os.path.join(root, 'frames')
        # Remove the 'frames' directory and all its contents
        try:
            shutil.rmtree(frames_dir_path)
            print(f"Removed {frames_dir_path}.")
        except Exception as e:
            print(f"Failed to remove {frames_dir_path}. Reason: {e}")

