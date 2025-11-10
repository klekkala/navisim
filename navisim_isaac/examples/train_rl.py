#!/usr/bin/env python3
"""
RL Training Example

Demonstrates how to train a navigation agent using the NaviSimIsaacEnv.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from navisim_isaac.gymnasium.env import NaviSimIsaacEnv


def random_agent_example():
    """
    Example: Random agent acting in the environment.

    This demonstrates the basic environment interface without actual training.
    """
    print("=" * 70)
    print("Random Agent Example")
    print("=" * 70)

    # Environment configuration
    config = {
        'max_episode_steps': 500,
        'max_linear_speed': 1.0,
        'max_angular_speed': 1.0,
        'camera_height': 84,
        'camera_width': 84,
        'headless': False,  # Set True for faster training
        'goal_tolerance': 0.5,
    }

    # Create environment
    graph_path = 'assets/sequence_graph.gpickle'

    try:
        env = NaviSimIsaacEnv(
            sequence_graph_path=graph_path,
            config=config,
            render_mode='human'
        )
    except FileNotFoundError:
        print(f"Error: Sequence graph not found at {graph_path}")
        print("Please create a sequence graph pickle file first.")
        return

    # Run episodes
    num_episodes = 3

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        observation, info = env.reset()

        print(f"Sector: {info['sector_id']}")
        print(f"Start: {info['start_position']}")
        print(f"Goal: {info['goal_position']}")

        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Random action
            action = env.action_space.sample()

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            done = terminated or truncated
            step += 1

            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}: "
                      f"Pos=({info['position'][0]:.2f}, {info['position'][1]:.2f}), "
                      f"Dist={info['distance_to_goal']:.2f}m, "
                      f"Reward={reward:.3f}")

        # Episode summary
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Reason: {info.get('termination_reason', 'unknown')}")

    # Cleanup
    env.close()
    print("\n✓ Example complete")


def simple_training_loop():
    """
    Example: Simple training loop with a policy.

    This shows how to integrate with RL algorithms (e.g., PPO, SAC).
    For actual training, use libraries like Stable-Baselines3.
    """
    print("=" * 70)
    print("Simple Training Loop Example")
    print("=" * 70)

    config = {
        'max_episode_steps': 200,
        'headless': True,  # Headless for faster training
        'camera_height': 84,
        'camera_width': 84,
    }

    graph_path = 'assets/sequence_graph.gpickle'

    try:
        env = NaviSimIsaacEnv(
            sequence_graph_path=graph_path,
            config=config,
            render_mode=None
        )
    except FileNotFoundError:
        print(f"Error: Sequence graph not found at {graph_path}")
        return

    # Training parameters
    num_episodes = 10

    # Simple statistics
    episode_rewards = []
    success_count = 0

    print(f"\nTraining for {num_episodes} episodes...")

    for episode in range(num_episodes):
        observation, info = env.reset()

        episode_reward = 0
        done = False

        while not done:
            # Simple policy: move forward and correct heading
            # In practice, replace with learned policy (neural network)

            # Get current state
            state = observation['state']
            pos_x, pos_y = state[0], state[1]
            yaw = state[7]

            # Get goal (from environment's goal_position)
            goal_x, goal_y = env.goal_position[0], env.goal_position[1]

            # Compute direction to goal
            dx = goal_x - pos_x
            dy = goal_y - pos_y
            target_yaw = np.arctan2(dy, dx)

            # Compute heading error
            heading_error = target_yaw - yaw
            # Normalize to [-pi, pi]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

            # Simple proportional control
            linear_vel = 0.8  # Move forward
            angular_vel = np.clip(heading_error * 2.0, -1.0, 1.0)  # Correct heading

            action = np.array([linear_vel, angular_vel], dtype=np.float32)

            # Step
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # Track statistics
        episode_rewards.append(episode_reward)
        if info.get('termination_reason') == 'goal_reached':
            success_count += 1

        # Print progress
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Avg Reward={avg_reward:.2f}, "
                  f"Success Rate={success_count}/{episode + 1}")

    # Final statistics
    print(f"\nTraining Complete!")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")

    env.close()


def stable_baselines3_example():
    """
    Example: Training with Stable-Baselines3.

    Requires: pip install stable-baselines3
    """
    print("=" * 70)
    print("Stable-Baselines3 Training Example")
    print("=" * 70)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("Error: stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        return

    # Environment configuration
    config = {
        'max_episode_steps': 500,
        'headless': True,
        'camera_height': 84,
        'camera_width': 84,
    }

    graph_path = 'assets/sequence_graph.gpickle'

    # Create vectorized environment
    def make_env():
        try:
            return NaviSimIsaacEnv(
                sequence_graph_path=graph_path,
                config=config,
                render_mode=None
            )
        except FileNotFoundError:
            print(f"Error: Sequence graph not found at {graph_path}")
            raise

    try:
        env = DummyVecEnv([make_env])
    except FileNotFoundError:
        return

    # Create PPO agent
    print("\nCreating PPO agent...")
    model = PPO(
        "MultiInputPolicy",  # For Dict observation space (image + state)
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./logs/navisim_ppo/"
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='navisim_ppo'
    )

    # Train
    print("\nStarting training...")
    print("Press Ctrl+C to stop")

    try:
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback
        )

        # Save final model
        model.save("navisim_ppo_final")
        print("\n✓ Training complete! Model saved to navisim_ppo_final.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save("navisim_ppo_interrupted")
        print("Model saved to navisim_ppo_interrupted.zip")

    env.close()


def test_trained_model():
    """
    Example: Test a trained model.
    """
    print("=" * 70)
    print("Test Trained Model")
    print("=" * 70)

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Error: stable-baselines3 not installed")
        return

    # Load model
    model_path = "navisim_ppo_final.zip"

    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        print("Train a model first using stable_baselines3_example()")
        return

    # Create environment
    config = {
        'max_episode_steps': 500,
        'headless': False,  # Show visualization
        'camera_height': 84,
        'camera_width': 84,
    }

    graph_path = 'assets/sequence_graph.gpickle'

    try:
        env = NaviSimIsaacEnv(
            sequence_graph_path=graph_path,
            config=config,
            render_mode='human'
        )
    except FileNotFoundError:
        print(f"Error: Sequence graph not found at {graph_path}")
        return

    # Test for multiple episodes
    num_test_episodes = 5

    for episode in range(num_test_episodes):
        print(f"\n--- Test Episode {episode + 1}/{num_test_episodes} ---")

        observation, info = env.reset()
        print(f"Sector: {info['sector_id']}")

        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Get action from trained model
            action, _states = model.predict(observation, deterministic=True)

            # Step
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1

            if step % 50 == 0:
                print(f"  Step {step}: Dist={info['distance_to_goal']:.2f}m")

        print(f"Episode finished: {info.get('termination_reason')}")
        print(f"  Steps: {step}, Reward: {episode_reward:.2f}")

    env.close()
    print("\n✓ Testing complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NaviSim-Isaac RL Training Examples")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'simple', 'sb3', 'test'],
        default='random',
        help='Example mode to run'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'random':
            random_agent_example()
        elif args.mode == 'simple':
            simple_training_loop()
        elif args.mode == 'sb3':
            stable_baselines3_example()
        elif args.mode == 'test':
            test_trained_model()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
