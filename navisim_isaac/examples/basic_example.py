"""
Basic Example Script

Demonstrates basic usage of NaviSim-Isaac integration.
"""

from navisim_isaac.gymnasium import NaviSimIsaacEnv, NavigationAgent, RLTrainer
from navisim_isaac.config import get_default_config
from navisim_isaac.utils import setup_logging, get_logger


def main():
    """
    Main function demonstrating basic usage.
    """
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)

    logger.info("Starting NaviSim-Isaac basic example")

    # Load configuration
    config = get_default_config()
    logger.info("Loaded configuration")

    # Create environment
    env = NaviSimIsaacEnv(config=config.get("gymnasium", {}).get("env", {}))
    logger.info("Created environment")

    # Create agent
    agent = NavigationAgent(config=config.get("gymnasium", {}).get("agent", {}))
    logger.info("Created agent")

    # Create trainer
    trainer = RLTrainer(
        env=env,
        agent=agent,
        config=config.get("gymnasium", {}).get("training", {})
    )
    logger.info("Created trainer")

    # Train (this will fail with NotImplementedError until implemented)
    try:
        logger.info("Starting training...")
        trainer.train(num_episodes=10)
    except NotImplementedError as e:
        logger.warning(f"Training not yet implemented: {e}")

    # Cleanup
    env.close()
    logger.info("Example completed")


if __name__ == "__main__":
    main()
