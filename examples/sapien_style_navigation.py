"""
Example usage of SAPIEN-style Navisim architecture.

This demonstrates how to use the new entity-component system
and SAPIEN-style environment patterns.
"""

import numpy as np
import navisim
from navisim import (
    NavigationEnv,
    NavisimScene, 
    Entity,
    TransformComponent,
    PhysicsComponent,
    SensorComponent,
    SequenceGraph,
    RenderMode
)


def main():
    """Demonstrate SAPIEN-style usage patterns."""
    
    print("🚀 SAPIEN-style Navisim Demo")
    print("=" * 50)
    
    # 1. Load assets using new asset management
    print("📦 Loading sequence graph...")
    try:
        sequence_graph = SequenceGraph("assets/sequence_graph.gpickle")
        print(f"✅ Loaded {len(sequence_graph.get_sequence_ids())} sequences")
    except FileNotFoundError:
        print("⚠️  Using mock sequence graph for demo")
        sequence_graph = create_mock_sequence_graph()
    
    # 2. Create SAPIEN-style environment
    print("\n🏗️  Creating navigation environment...")
    env = NavigationEnv(
        sequence_graph=sequence_graph,
        render_mode="rgb_array",
        control_freq=60
    )
    print(f"✅ Environment created: {env.__class__.__name__}")
    
    # 3. Demonstrate entity-component system
    print("\n🔧 Demonstrating Entity-Component System...")
    demonstrate_ecs(env.scene)
    
    # 4. Run episode using SAPIEN patterns
    print("\n🎮 Running navigation episode...")
    run_navigation_episode(env)
    
    # 5. Cleanup
    env.close()
    print("\n✅ Demo completed successfully!")


def create_mock_sequence_graph():
    """Create a mock sequence graph for demo purposes."""
    class MockSector:
        def __init__(self, seq_id, sector_id):
            self.seq_id = seq_id
            self.sector_id = sector_id
        
        def random_spawn_pose(self):
            return [0, 0, 0, 0, 0, 0]  # x, y, z, roll, pitch, yaw
        
        def get_target_locations(self):
            return np.array([[5.0, 5.0], [10.0, 10.0]], dtype=np.float32)
        
        def load(self):
            pass
        
        def unload_all(self):
            pass
    
    class MockSequenceGraph:
        def get_sequence_ids(self):
            return ["mock_sequence_1", "mock_sequence_2"]
        
        def get_sequence(self, seq_id):
            return [MockSector(seq_id, 0)]
    
    return MockSequenceGraph()


def demonstrate_ecs(scene: NavisimScene):
    """Demonstrate entity-component system usage."""
    
    # Create a custom entity with multiple components
    robot_entity = Entity(name="DemoRobot")
    
    # Add transform component
    initial_pose = np.eye(4, dtype=np.float64)
    initial_pose[:3, 3] = [1.0, 2.0, 0.0]  # Position at (1, 2, 0)
    robot_entity.add_component(TransformComponent(pose=initial_pose))
    
    # Add physics component
    robot_entity.add_component(PhysicsComponent(mass=50.0, friction=0.8))
    
    # Add camera sensor
    camera_config = {
        "width": 640,
        "height": 480,
        "fov": 90.0
    }
    robot_entity.add_component(SensorComponent(
        sensor_type="camera",
        update_frequency=30.0,
        sensor_config=camera_config
    ))
    
    # Add to scene
    scene.add_entity(robot_entity)
    
    print(f"  📍 Created entity: {robot_entity}")
    print(f"  🔧 Components: {list(robot_entity.components.keys())}")
    
    # Demonstrate component interaction
    transform = robot_entity.get_component(TransformComponent)
    physics = robot_entity.get_component(PhysicsComponent)
    sensor = robot_entity.get_component(SensorComponent)
    
    print(f"  📍 Initial position: {transform.get_position()}")
    
    # Apply force and simulate
    physics.apply_force(np.array([10.0, 0.0, 0.0]))
    
    # Step simulation
    for i in range(5):
        robot_entity.step(1/60.0)  # 60 Hz
        if i == 2:
            print(f"  📍 Position after {i+1} steps: {transform.get_position()}")
    
    print(f"  📍 Final position: {transform.get_position()}")
    print(f"  📊 Sensor data shape: {sensor.get_latest_data().shape if sensor.get_latest_data() is not None else 'None'}")


def run_navigation_episode(env: NavigationEnv):
    """Run a navigation episode using SAPIEN patterns."""
    
    # Reset environment following SAPIEN patterns
    observation, info = env.reset(seed=42)
    print(f"  🔄 Environment reset")
    print(f"  📊 Observation keys: {list(observation.keys())}")
    print(f"  🎯 Targets: {info.get('num_targets', 0)}")
    print(f"  📏 Distance to target: {info.get('distance_to_target', 'unknown'):.2f}")
    
    total_reward = 0.0
    max_steps = 50
    
    for step in range(max_steps):
        # Choose action (simple policy: mostly forward with occasional turns)
        if step % 10 == 0 and step > 0:
            action = np.random.choice([2, 3])  # Turn left or right
        else:
            action = 1  # Move forward
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print progress
        if step % 10 == 0 or terminated:
            print(f"  📈 Step {step}: reward={reward:.3f}, distance={info.get('distance_to_target', 0):.2f}")
        
        if terminated:
            print(f"  🎉 Episode completed! Target reached in {step+1} steps")
            break
        
        if truncated:
            print(f"  ⏰ Episode truncated at step {step+1}")
            break
    
    print(f"  📊 Total reward: {total_reward:.3f}")
    print(f"  📸 Final observation shape: {observation['rgb'].shape}")


if __name__ == "__main__":
    main()