import tarfile
import os
import json
import glob

# Assuming 'tar_files_path' is the directory containing your .tar.gz files
tar_files_path = '/home/ec2-user'

# List of your .tar.gz files
tar_files = glob.glob(os.path.join(tar_files_path, "*.tar.gz"))

# Iterate and extract each tar.gz file
for tar_file in tar_files:
    # Full path to the current tar.gz file is already provided by glob
    tar_file_path = tar_file

    # Extract the tar.gz file
    with tarfile.open(tar_file_path, 'r:gz') as tar:
        extract_dir = os.path.join(tar_files_path, tar_file.replace('.tar.gz', ''))
        os.makedirs(extract_dir, exist_ok=True)
        tar.extractall(path=extract_dir)
    # Delete the original .tar.gz file
    os.remove(tar_file_path)        

