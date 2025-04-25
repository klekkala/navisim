import logging
import argparse

from pathlib import Path
# from beogym.beogym import BeoGym
from data_loader import *
from util.logging_util import logging_setup


def parse_agent_args(parser):
    #TODO(Haopeng) : Add arguments for the parameters to be used for the agent
    return parser

def parse_env_ags(parser):
    parser.add_argument('--seq_graph', type = str, default = "sequence_graph.gpickle")
    parser.add_argument('--database', type = str, default = 'rocksdb')
    return parser

def check_env(database = 'rocksdb', sq_graph = 'sequence_graph.gpickle'):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    # Checking for necessary set up files
    assets_folder = Path(project_root, 'assets')
    sq_graph_path = Path(assets_folder, sq_graph)
    database_path = Path(assets_folder, database)

    if not sq_graph_path.is_file():
        raise FileNotFoundError(f"Required sequence graph not found: {sq_graph_path}")
    
    if not database_path.is_dir():
        raise FileNotFoundError(f"Required database not found: {database_path}")
    
    return sq_graph_path, database_path

class NavisimMaster:
    def __init__(self):
        return
    
    def run():
        return

if __name__ == "__main__":
    logging_setup(logging.INFO)

    parser = argparse.ArgumentParser(description='Navisim')
    parser = parse_agent_args(parser)
    parser = parse_env_ags(parser)
    args = parser.parse_args()

    seq_path, database_path = check_env(database = args.database, sq_graph = args.seq_graph)
    
    logging.info('**** Navisim Starting ****')
    sequence_graph = load_sequence_graph(seq_path)
    NavisimMaster()

    




