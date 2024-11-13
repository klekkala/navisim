import os
import shutil
import argparse

def copy_sector_lego(data_sector, log_file):
    try:
        destination_path = os.path.join(args.target, data_sector, "sector_lego")
        os.makedirs(destination_path, exist_ok=True)

        # source_sparse_path = os.path.join(args.source, data_sector, "sparse")
        # if os.path.exists(source_sparse_path):
        #     for item in os.listdir(source_sparse_path):
        #         source_item = os.path.join(source_sparse_path, item)
        #         destination_item = os.path.join(destination_path, item)

        #         if os.path.isdir(source_item):
        #             shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
        #             print(f"Copied directory {source_item} to {destination_item}")
        #         else:
        #             shutil.copy(source_item, destination_item)
        #             print(f"Copied file {source_item} to {destination_item}")

        # Copy all pcd files from bag_dump
        source_path = os.path.join(args.source, data_sector)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path does not exist for {data_sector}.")

        for item in os.listdir(source_path):
            source_item = os.path.join(source_path, item)
            destination_item = os.path.join(destination_path, item)

            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
                print(f"Copied directory {source_item} to {destination_item}")
            else:
                shutil.copy(source_item, destination_item)
                print(f"Copied file {source_item} to {destination_item}")
    
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error copying sector {data_sector}: {str(e)}\n")
        print(f"Logged error for {data_sector}")

def list_subfolders(directory, log_file):
    try:
        all_items = os.listdir(directory)
        subfolders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
        subfolders.sort()

        for subfolder in subfolders:
            copy_sector_lego(subfolder, log_file)

    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error listing subfolders in {directory}: {str(e)}\n")
        print(f"Logged error for listing subfolders in {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Source directory')
    parser.add_argument('--target', type=str, required=True, help='Target directory')
    args = parser.parse_args()
    
    log_file = "sector_lego_error_log.txt"
    list_subfolders(args.target, log_file)


# python3 get_sector_lego.py --source "/lab/tmpig23b/navisim/data/bag_dump/2023_03_11/0/all_lego" --target "/lab/tmpig23b/navisim/data/gs_train/2023_03_11/0"
# python3 get_sector_lego.py --source "/lab/tmpig23b/navisim/data/bag_dump/2023_03_11/0/all_lego" --target "/lab/tmpig23b/navisim/data/gs_train_test_three/2023_03_11/0"