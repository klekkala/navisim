# scripts/run.py

import argparse
from isaaclab.app import AppLauncher


def main():
    # ---------------------------------------------------
    # 1. Parse arguments (DO NOT add --headless manually)
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(description="Run Navisim with Isaac Lab.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)


    # AppLauncher injects: --headless, --no-window, --device, etc.
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # ---------------------------------------------------
    # 2. Launch Isaac Lab / SimulationApp (the RIGHT way)
    # ---------------------------------------------------
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app   # <-- this *is* SimulationApp

    # ---------------------------------------------------
    # 3. Import IsaacLab and Navisim modules AFTER launch
    # ---------------------------------------------------
    from tasks.navigation_task import NavigationTask

    # ---------------------------------------------------
    # 4. Create your RL task
    # ---------------------------------------------------
    task = NavigationTask(
        num_envs=args.num_envs,
        max_episode_len=args.max_steps,
    )

    obs = task.reset()
    t = 0

    print(f"[Navisim] Starting simulation. Headless: {args.headless}")

    # ---------------------------------------------------
    # 5. Main simulation loop
    # ---------------------------------------------------
    while simulation_app.is_running():

        # Example: forward 100 steps, turn 20 steps
        if t % 120 < 100:
            actions = task.sample_forward_action()
        else:
            actions = task.sample_turn_action()

        obs, rew, done, _ = task.step(actions)

        if done.any():
            task.reset()

        t += 1

    # ---------------------------------------------------
    # 6. Cleanly close Isaac Sim
    # ---------------------------------------------------
    simulation_app.close()


if __name__ == "__main__":
    main()
