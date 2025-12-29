
import gymnasium as gym
import gymnasium as gym
import torch
import numpy as np
import random
import logging
from pernaf.pernaf.agent import NAF

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Seed
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Standard Gymnasium Environment
    env = gym.make("Pendulum-v1")
    
    # Configuration
    config = dict(
        learning_rate=1e-3,
        batch_size=100,
        buffer_size=100000,
        discount=0.99,
        polyak=0.999,
        max_steps=200,
        update_repeat=5,
        # Slower noise decay: 1 / (1 + episode * 0.01)
        # At ep 100: 1/2 = 0.5. At ep 200: 1/3 = 0.33. 
        noise_info=dict(noise_function=lambda nr: 1.0 / (1.0 + nr * 0.02))
    )
    
    # NAF Agent
    agent = NAF(env, **config)
    
    # Train
    logger.info("Starting training on Pendulum-v1 (Improved)...")
    agent.learn(total_timesteps=50000)
    logger.info("Training finished.")
    
    # Save
    agent.save("checkpoints/pendulum_model")
    
    # Plotting
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # The agent tracks losses and values internally
    if hasattr(agent, 'losses') and hasattr(agent, 'vs'):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(agent.losses, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(agent.vs, label='Value Est')
        plt.title('Value Estimates')
        plt.xlabel('Updates')
        plt.ylabel('V')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/pendulum_training_stats.png')
        logger.info("Saved training stats to results/pendulum_training_stats.png")
        
        # Save raw data
        df = pd.DataFrame({'loss': agent.losses, 'value': agent.vs})
        df.to_csv('results/pendulum_stats.csv', index=False)
        
    # Evaluate
    obs, info = env.reset(seed=seed)
    total_reward = 0
    rewards = []
    
    # Run one full episode for visualization (Pendulum is 200 steps)
    states = []
    actions = []
    
    for _ in range(200):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        states.append(obs)
        actions.append(action)
        if terminated or truncated:
            break
            
    logger.info(f"Evaluation Reward: {total_reward}")
    
    # Plot Evaluation
    states = np.array(states)
    actions = np.array(actions)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Evaluation: Rewards')
    plt.ylabel('Reward')
    
    plt.subplot(3, 1, 2)
    plt.plot(states[:, 0], label='cos(theta)')
    plt.plot(states[:, 1], label='sin(theta)')
    plt.plot(states[:, 2], label='theta_dot')
    plt.legend()
    plt.title('Evaluation: States')
    
    plt.subplot(3, 1, 3)
    plt.plot(actions)
    plt.title('Evaluation: Actions')
    plt.ylabel('Torque')
    plt.xlabel('Step')
    
    plt.tight_layout()
    plt.savefig('results/pendulum_eval_plot.png')
    logger.info("Saved eval plot to results/pendulum_eval_plot.png")
    
    env.close()

if __name__ == "__main__":
    main()
