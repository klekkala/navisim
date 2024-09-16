import logging
import argparse
from util.logging_util import logging_setup

def parse_agent_args(parser):
    #TODO(Haopeng) : Add arguments for the parameters to be used for the agent
    return parser

def parse_env_ags(parser):
    parser.add_argument('-r', '--resolution', default = 10, help = 'Resolution for the elevation map')
    return parser

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

    logging.info('**** Navisim Starting ****')



