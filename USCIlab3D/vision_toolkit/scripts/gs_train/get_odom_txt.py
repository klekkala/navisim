import os
import shutil
import argparse

def copy_odometry(data_sector, log_file):
    odom_folder = "all_odom"
    file_to_copy = "odometry.txt"
    
    try:
        destination_path = os.path.join(args.target, data_sector)
        if not os.path.exists(destination_path):
            raise FileNotFoundError(f"gs_train path doesn't exist for {data_sector}.")

        source_path = os.path.join(args.source, odom_folder, data_sector, file_to_copy)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"bag_dump path/file doesn't exist for {data_sector}.")

        shutil.copy(source_path, destination_path)
        print(f"Copied {source_path} to {destination_path}")
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error copying {data_sector}: {str(e)}\n")
        print(f"Logged error for {data_sector}")

def list_subfolders(directory, log_file):
    try:
        all_items = os.listdir(directory)
        subfolders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
        subfolders.sort()

        for subfolder in subfolders:
            copy_odometry(subfolder, log_file)

    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error listing subfolders in {directory}: {str(e)}\n")
        print(f"Logged error for listing subfolders in {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Source directory')
    parser.add_argument('--target', type=str, required=True, help='Target directory')
    args = parser.parse_args()
    
    log_file = "odom_error_log.txt"
    list_subfolders(args.target, log_file)


# python3 get_odom_txt.py --source "/lab/tmpig23b/navisim/data/bag_dump/2023_03_11/0/" --target "/lab/tmpig23b/navisim/data/gs_train/2023_03_11/0"
# python3 get_odom_txt.py --source "/lab/tmpig23b/navisim/data/bag_dump/2023_03_11/0/" --target "/lab/tmpig23b/navisim/data/gs_train_test_three/2023_03_11/0"
