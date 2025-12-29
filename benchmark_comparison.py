
import gymnasium as gym
import torch
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pernaf.pernaf.agent import NAF

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_experiment(name, config, seed=123, steps=40000):
    logging.info(f"--- Starting {name} ---")
    set_seed(seed)
    
    env = gym.make("Pendulum-v1")
    agent = NAF(env, **config)
    
    # Train
    agent.learn(total_timesteps=steps)
    
    # Return stats
    return agent.episode_rewards

def moving_average(data, window_size=50):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

def main():
    logging.basicConfig(level=logging.INFO)
    
    steps = 40000
    
    # Tuned noise function (slower decay)
    tuned_noise = dict(noise_function=lambda nr: 1.0 / (1.0 + nr * 0.02))

    # 1. NAF (Standard)
    # Single Q, Uniform Replay
    config_naf = dict(
        learning_rate=1e-3, batch_size=100, buffer_size=100000,
        discount=0.99, polyak=0.999, update_repeat=5,
        prio_info={},
        double_q=False,
        noise_info=tuned_noise
    )
    
    # 2. NAF2 (Double Q)
    # Double Q, Uniform Replay
    config_naf2 = dict(
        learning_rate=1e-3, batch_size=100, buffer_size=100000,
        discount=0.99, polyak=0.999, update_repeat=5,
        prio_info={},
        double_q=True,
        noise_info=tuned_noise
    )
    
    # 3. PER-NAF2 (Double Q + PER)
    # Double Q, Prioritized Replay
    config_per_naf2 = dict(
        learning_rate=1e-3, batch_size=100, buffer_size=100000,
        discount=0.99, polyak=0.999, update_repeat=5,
        prio_info=dict(alpha=0.6, beta=0.4),
        double_q=True,
        noise_info=tuned_noise
    )

    # Run
    rewards_naf = run_experiment("NAF (Single Q)", config_naf, steps=steps)
    rewards_naf2 = run_experiment("NAF2 (Double Q)", config_naf2, steps=steps)
    rewards_per_naf2 = run_experiment("PER-NAF2 (Double Q + PER)", config_per_naf2, steps=steps)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.plot(moving_average(rewards_naf), label='NAF (Single Q)', alpha=0.7)
    plt.plot(moving_average(rewards_naf2), label='NAF2 (Double Q)', alpha=0.7)
    plt.plot(moving_average(rewards_per_naf2), label='PER-NAF2 (Double Q + PER)', alpha=0.7)
    
    plt.title(f'Performance Comparison (Rewards) on Pendulum-v1 ({steps} steps)')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/naf_vs_naf2_rewards.png')
    logging.info("Saved comparison plot to results/naf_vs_naf2_rewards.png")

if __name__ == "__main__":
    main()
