import subprocess
import os
import sys

# Paths
SECTOR_PATH = ""
DATABASE_PATH = ""
IMAGE_PATH = ""
SPARSE_PATH = ""
DENSE_PATH = ""


def run_colmap_feature_extractor():
    """Runs the COLMAP feature extractor."""
    print("Path")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGE_PATH,
    ], check=True)
    
def run_colmap_exhaustive_matcher():
    """Runs the COLMAP exhaustive matcher."""
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", DATABASE_PATH
    ], check=True)

def run_colmap_mapper():
    """Runs the COLMAP mapper for sparse reconstruction."""
    subprocess.run([
        "colmap", "mapper",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGE_PATH,
        "--output_path", SPARSE_PATH
    ], check=True)

def run_colmap_image_undistorter():
    """Runs the COLMAP image undistorter for dense reconstruction."""
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", IMAGE_PATH,
        "--input_path", os.path.join(SPARSE_PATH, "0"),
        "--output_path", DENSE_PATH,
        "--output_type", "COLMAP"
    ], check=True)

def run_colmap_patch_match_stereo():
    """Runs the COLMAP patch_match_stereo for dense reconstruction."""
    subprocess.run([
        "colmap", "patch_match_stereo",
        "--workspace_path", DENSE_PATH,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ], check=True)

def run_colmap_stereo_fusion():
    """Runs the COLMAP stereo fusion."""
    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", DENSE_PATH,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",  # You can change this to "photometric" if you don't want geometric consistency
        "--output_path", os.path.join(DENSE_PATH, "fused.ply")
    ], check=True)

def run_colmap_poisson_mesher():
    """Runs Poisson surface reconstruction to generate meshed-poisson.ply."""
    subprocess.run([
        "colmap", "poisson_mesher",
        "--input_path", os.path.join(DENSE_PATH, "fused.ply"),
        "--output_path", os.path.join(DENSE_PATH, "meshed-poisson.ply")
    ], check=True)

def run_colmap_delaunay_mesher():
    """Runs Delaunay meshing to generate meshed-delaunay.ply."""
    subprocess.run([
        "colmap", "delaunay_mesher",
        "--input_path", DENSE_PATH,
        "--input_type", "dense",
        "--output_path", os.path.join(DENSE_PATH, "meshed-delaunay.ply")
    ], check=True)



if __name__ == "__main__":
    SECTOR_PATH = sys.argv[1]
    DATABASE_PATH = f"{SECTOR_PATH}/colmap/database.db"
    IMAGE_PATH = f"{SECTOR_PATH}/input"
    SPARSE_PATH = f"{SECTOR_PATH}/colmap/sparse"
    DENSE_PATH = f"{SECTOR_PATH}/colmap/dense"

    # Create directories / files if they don't exist
    os.makedirs(SPARSE_PATH, exist_ok=True)
    # os.makedirs(DENSE_PATH, exist_ok=True)
    if not os.path.exists(DATABASE_PATH):
        open(DATABASE_PATH, 'a').close()

    # Run the COLMAP pipeline step by step
    try:
        print("Running feature extraction...")
        run_colmap_feature_extractor()
        
        print("Running exhaustive matcher...")
        run_colmap_exhaustive_matcher()
        
        print("Running sparse reconstruction (mapper)...")
        run_colmap_mapper()
        
        print("Running image undistortion...")
        run_colmap_image_undistorter()
        
        print("Running dense reconstruction (patch match stereo)...")
        run_colmap_patch_match_stereo()
        
        print("Running stereo fusion...")
        run_colmap_stereo_fusion()    

        #------------------------------------------------------------------------
        # Mesher Functions
        # print("Running Poisson Mesher...")
        # run_colmap_poisson_mesher()

        # print("Running Delaunay Mesher...")
        # run_colmap_delaunay_mesher()
        #------------------------------------------------------------------------

        print("All Done")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


# Example: python3 ./get_colmap.py /lab/tmpig23b/navisim/data/gs_train_test_three/2023_03_11/0/sector2