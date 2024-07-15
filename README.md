# navisim
Gausian splatting based rendering
------
## Directory Structure
```
project
│   README.md
├─── assets/ (any resource files)
│    ├── gaussian_splat_data/ 
│    │   ├── sector_{n}/
│    │   │   ├── point_cloud_data_file.ply
│    │   │   ├── transformation_matrix.txt
│    │   │   └── camera.json
│    │   └── .../
│    └── .../
├─── output/
│    ├── elevation.png
│    └── occupancy.png
├─── submodule/
│    └── gaussian/
│          └── ...
├─── src/
│    ├── main.py
│    ├── paths.py (contains path to all files)
│    │ ....
│    ├─── agent/
│    │   └── agent.py
│    ├─── beogym/
│    │   ├── pointcloud/
│    │   │    ├──  pointcloud.py
│    │   │    └─ ...
│    │   └─ ...
│    │   ├── sequence_graph/
│    │   │    ├──  sequence_graph.py
│    │   │    └─ ...
│    │   └── ...
│    └── ...
├─── tests/   (optional)
│    ├─── test_env.py
│    └── ...    
└─── evaluation/
       └── ...   
```
