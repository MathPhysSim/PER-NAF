
import os
import sys
import tensorflow as tf
import numpy as np
import random
import logging

from pernaf.pernaf.agent import NAF
from simulated_environment import AwakeElectronEnv


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parameters
    random_seed = 123
    try:
        if len(sys.argv) > 1:
            random_seed = int(sys.argv[1])
    except ValueError:
        pass
        
    logger.info(f"Starting NAF training with seed {random_seed}")
    
    # Set seeds
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Initialize Environment
    # Note: Environment expects 'electron_tt43.out' to be available
    try:
        env = AwakeElectronEnv()
    except FileNotFoundError as e:
        logger.error(e)
        logger.error("Please ensure 'electron_tt43.out' is present.")
        sys.exit(1)

    # Training Configuration
    directory = f'checkpoints/naf_seed_{random_seed}'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    discount = 0.999
    batch_size = 10
    learning_rate = 1e-3
    max_steps = 200 # Shortened for testing, original was higher
    update_repeat = 3
    max_episodes = 50 # Shortened for testing
    
    # Prioritized Experience Replay (PER) Configuration
    # prio_info = dict(alpha=.5, beta_start=.9, beta_decay=lambda nr: max(1e-16, 0.25*(1 - nr / 100)))
    prio_info = dict() # Default to no PER for base run, uncomment above to enable
    
    noise_info = dict(noise_function=lambda nr: max(0, 2*(1 - nr / 500)))
    
    nafnet_kwargs = dict(
        hidden_sizes=[100, 100], 
        activation=tf.nn.tanh,
        learning_rate=learning_rate
    )

    # Initialize Agent
    # Refactored to SB3 Style
    
    agent = NAF(
        env=env,
        discount=discount,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        update_repeat=update_repeat,
         # max_episodes was removed in favor of total_timesteps in learn()
        prio_info=prio_info,
        noise_info=noise_info,
        directory=directory,
        **nafnet_kwargs
    )
    
    # Run Training
    logger.info("Starting run...")
    # Convert max_episodes to approximate total_timesteps for the learn method
    total_timesteps = max_episodes * max_steps
    agent.learn(total_timesteps=total_timesteps)
    logger.info("Run finished.")
    
    # Save final model
    agent.q_target_model_1.save_model(os.path.join(directory, 'final_model'))

if __name__ == '__main__':
    main()
