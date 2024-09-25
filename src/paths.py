import os 

_current_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(_current_dir, "../"))
_assets_folder = os.path.join(_root_dir, 'assets')
_cache_folder = os.path.join(_root_dir, 'cache')

GAUSSIAN_SPLAT_FOLDER = os.path.join(_assets_folder, 'gaussian_splat_data')
SEQUENCE_GRAPH_FOLDER = os.path.join(_cache_folder, 'sequence_graph')
