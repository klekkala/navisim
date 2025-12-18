#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import gymnasium as gym
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # wrapper must be last :contentReference[oaicite:6]{index=6}

# This import must happen BEFORE gym.make() so the task is registered.
import navisim_lab.tasks  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Navisim-Warehouse-Jetbot-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--enable_cameras", action="store_true", help="Needed if you use camera sensors.")
    parser.add_argument(
        "--agent_cfg",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "navisim_lab" / "configs" / "rsl_rl" / "ppo_warehouse_jetbot.yaml"),
    )
    parser.add_argument("--log_dir", type=str, default="../logs/rsl_rl")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Launch Isaac Sim
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=args.enable_cameras,
    )
    simulation_app = app_launcher.app

    # 2) Make env
    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        seed=args.seed,
        # If your env constructor supports these, pass through:
        # headless=args.headless,
        # enable_cameras=args.enable_cameras,
    )

    # 3) Load agent config (YAML) and convert to dict
    agent_cfg = OmegaConf.load(args.agent_cfg)
    agent_cfg.seed = args.seed  # keep CLI seed authoritative
    agent_cfg_dict = OmegaConf.to_container(agent_cfg, resolve=True)

    # 4) Wrap env for RSL-RL (last wrapper)
    clip_actions = agent_cfg_dict.get("clip_actions", None)
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    # 5) Create runner
    from rsl_rl.runners import OnPolicyRunner  # from leggedrobotics/rsl_rl package

    log_dir = Path(args.log_dir) / agent_cfg_dict.get("experiment_name", "exp") / agent_cfg_dict.get("run_name", "run")
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=str(log_dir))

    # 6) Train
    max_iters = int(agent_cfg_dict["max_iterations"])
    runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)

    # 7) Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
