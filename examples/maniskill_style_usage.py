"""
Example usage of Navisim with ManiSkill-style interface.

This demonstrates the new task-oriented architecture following
ManiSkill patterns for environment creation and agent interaction.
"""

import numpy as np
import navisim

def main():
    """Demonstrate ManiSkill-style usage."""
    
    # 1. Create environment using ManiSkill-style registration
    print("Creating navigation environment...")
    env = navisim.make_env(
        "NavisimNavigation-v1",
        obs_mode="state_dict",
        render_mode="rgb_array",
        max_episode_steps=500
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 2. Run episode with random actions
    print("\nRunning episode with random actions...")
    obs, info = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Agent position: {obs['agent_pos']}")
    print(f"Goal position: {obs['goal_pos']}")
    
    total_reward = 0
    for step in range(100):
        # Random action: [linear_velocity, angular_velocity]
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, goal_distance={info['goal_distance']:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Success: {info['success']}")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    
    # 3. Demonstrate agent usage independently
    print("\nDemonstrating standalone agent usage...")
    
    # Create a navigation agent
    agent = navisim.NavigationAgent(
        scene=None,  # Would normally pass actual scene
        name="test_agent",
        max_speed=2.0,
        sensor_range=10.0
    )
    
    # Set initial position and goal
    agent.set_position(np.array([0.0, 0.0, 0.5]))
    goal_pos = np.array([5.0, 5.0, 0.5])
    
    print(f"Agent initial position: {agent.get_position()}")
    print(f"Goal position: {goal_pos}")
    
    # Simulate navigation
    for i in range(10):
        # Get navigation command
        desired_vel = agent.navigate_to_goal(goal_pos)
        agent.set_velocity(desired_vel)
        
        # Step agent
        agent.step(dt=0.1)
        
        current_pos = agent.get_position()
        distance = np.linalg.norm(current_pos - goal_pos)
        
        if i % 3 == 0:
            print(f"Step {i}: pos={current_pos}, distance={distance:.3f}")
        
        if distance < 0.1:
            print("Goal reached!")
            break
    
    # 4. Show sensor data
    print("\nAgent sensor data:")
    sensor_data = agent.get_sensor_data()
    for sensor_name, data in sensor_data.items():
        print(f"{sensor_name}: shape={data.shape}, mean={np.mean(data):.3f}")
    
    # Clean up
    env.close()
    print("\nExample completed successfully!")


def demonstrate_environment_registration():
    """Show how to register custom environments."""
    
    @navisim.register_env("CustomNav-v1", max_episode_steps=200)
    class CustomNavigationTask(navisim.BaseEnv):
        """Custom navigation task with modified reward."""
        
        def _build_world(self):
            # Custom world setup
            pass
        
        def _setup_spaces(self):
            # Custom action/observation spaces
            self.action_space = navisim.BaseEnv.action_space
            self.observation_space = navisim.BaseEnv.observation_space
        
        def _get_obs(self):
            return {"state": np.zeros(4)}
        
        def _get_info(self):
            return {"custom_info": True}
        
        def _reset_simulation(self):
            pass
        
        def _apply_action(self, action):
            pass
        
        def _step_simulation(self):
            pass
        
        def _compute_reward(self):
            return 1.0  # Custom reward
    
    print(f"Registered environments: {list(navisim.envs.REGISTERED_ENVS.keys())}")
    
    # Create custom environment
    custom_env = navisim.make_env("CustomNav-v1")
    print(f"Created custom environment: {custom_env.__class__.__name__}")


if __name__ == "__main__":
    print("=== Navisim ManiSkill-Style Usage Example ===\n")
    
    try:
        main()
        print("\n" + "="*50)
        demonstrate_environment_registration()
    except Exception as e:
        print(f"Error running example: {e}")
        print("Note: Some features require full simulation backend setup")