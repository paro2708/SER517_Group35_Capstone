import os
import json
import glob
import boto3

s3_client = boto3.client('s3')

base_directory_path = '/home/ec2-user'

info_files_pattern = os.path.join(base_directory_path, '*', '*/info.json')

info_files = glob.glob(info_files_pattern)

for info_file_path in info_files:
    try:
        with open(info_file_path, 'r') as file:
            info_data = json.load(file)
            dataset_type = info_data.get('Dataset', '').lower()
    except FileNotFoundError:
        print(f"File {info_file_path} not found.")
        continue

    # Determine the bucket based on the dataset type
    bucket_name = "train-capstone-35" if dataset_type == "train" else "test-capstone-35" if dataset_type == "test" else None

    if bucket_name:
        directory_to_upload = os.path.dirname(info_file_path)

        for root, dirs, files in os.walk(directory_to_upload):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = os.path.relpath(file_path, start=base_directory_path)
                
                # Upload the file to the corresponding S3 bucket
                try:
                    s3_client.upload_file(file_path, bucket_name, s3_key)
                    print(f"Uploaded {file_path} to {bucket_name}/{s3_key}.")
                except Exception as e:
                    print(f"Failed to upload {file_path} to {bucket_name}/{s3_key}. Reason: {e}")
    else:
        print(f"Dataset type for {info_file_path} is not recognized. Skipping upload.")


