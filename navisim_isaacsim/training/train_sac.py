# training/train_sac.py

import yaml
import argparse

from isaaclab.envs import make
from isaaclab_rl.rlgames.train_rlgames import run_rlgames


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="training/sac_config.yaml")
    args = parser.parse_args()

    # Load YAML training confi
    config = load_yaml(args.cfg)

    # 1. Create IsaacLab environment
    env_cfg = config["env"]
    env = make("NavisimNavigationEnv", cfg=env_cfg, auto_reset=True)

    # 2. Inject dimensions into network config
    config["network"]["mlp"]["input_size"] = env.obs_dim
    config["network"]["mlp"]["output_size"] = env.act_dim

    # 3. Run RL-Games inside Isaac Lab
    print("[SAC] Starting SAC training...")
    run_rlgames(env, config)
    print("[SAC] Training finished!")


if __name__ == "__main__":
    main()
