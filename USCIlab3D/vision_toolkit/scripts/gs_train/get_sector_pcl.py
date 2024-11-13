import os
import shutil
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--target', type=str)
args = parser.parse_args()

pcl_source_folder = args.data
pcl_target_folder_base = args.target

def copy_file(source_path, destination_path):
    if not os.path.exists(destination_path):
        try:
            shutil.copy2(source_path, destination_path)
            print(f"File copied successfully from '{source_path}' to '{destination_path}'.")
        except FileNotFoundError:
            log_error(f"Error: The file '{source_path}' does not exist.")
        except PermissionError:
            log_error(f"Error: Permission denied. Unable to copy the file.")

def log_error(message):
    # Open the error log file in append mode and write the error message
    with open('error_log.txt', 'a') as log_file:
        log_file.write(message + '\n')

def find_matching_folder(target_folder, prefix):
    cam_prefixes = ["cam1_", "cam2_", "cam3_", "cam4_", "cam5_"]
    
    try:
        # Search in the target folder and all subfolders
        for root, dirs, files in os.walk(target_folder):
            # Check for files that match the pattern camX_prefix.*
            for file in files:
                for cam_prefix in cam_prefixes:
                    if file.startswith(f"{cam_prefix}{prefix}"):
                        return root  # Return the folder path where the match is found
    except Exception as e:
        log_error(f"Error while searching for prefix {prefix}: {str(e)}")
    return None  # If no match is found, return None


if __name__ == "__main__":
    # Iterate through all files in the source folder
    for filename in os.listdir(pcl_source_folder):
        if filename.endswith('.pcd'):  # Process only PCD files
            try:
                # Extract the prefix (before the first dot)
                prefix = filename.split('.')[0]
                matching_folder = find_matching_folder(pcl_target_folder_base, prefix)

                if matching_folder:
                    parent_folder = os.path.dirname(matching_folder)
                    pcl_target_folder = os.path.join(parent_folder, "sector_pcl")
                    if not os.path.exists(pcl_target_folder):
                        os.makedirs(pcl_target_folder)
                
                    source_file = os.path.join(pcl_source_folder, filename)
                    target_file = os.path.join(pcl_target_folder, filename)
                    copy_file(source_file, target_file)
                else:
                    log_error(f"No matching folder found for {filename}")

            except Exception as e:
                    # Log the error if any exception occurs during file processing
                    log_error(f"Error processing file {filename}: {str(e)}")


# python3 ./get_sector_pcl.py --data "/lab/tmpig23b/navisim/data/bag_dump/2023_03_11/0/all_pcl" --target "/lab/tmpig23b/navisim/data/gs_train/2023_03_11/0"
# python3 ./get_sector_pcl.py --data "/lab/tmpig23b/navisim/data/bag_dump/2023_03_11/0/all_pcl" --target "/lab/tmpig23b/navisim/data/gs_train_test_three/2023_03_11/0"
