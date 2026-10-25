import logging
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pernaf.pernaf.agent import NAF


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make("Pendulum-v1")

    config = dict(
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=50000,
        discount=0.98,
        polyak=0.995,
        max_steps=200,
        update_repeat=2,
        warmup_steps=400,
        net_arch=[64, 64],
        prio_info=dict(alpha=0.6, beta=0.4),
        double_q=True,
        noise_info=dict(noise_function=lambda nr: max(0.05, 1.0 / (1.0 + nr * 0.08))),
        seed=seed,
    )

    agent = NAF(env, **config)

    logger.info("Starting training on Pendulum-v1 (PER-NAF2 JAX+Numba)...")
    agent.learn(total_timesteps=20000)
    logger.info("Training finished.")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    agent.save("checkpoints/pendulum_model.pt")

    if hasattr(agent, "losses") and hasattr(agent, "vs"):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(agent.losses, label="Huber TD Loss", color="#1f77b4", alpha=0.8)
        plt.title("Training Loss (PER-NAF2 on Pendulum-v1)")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(agent.vs, label="Value Estimate V(s)", color="#2ca02c", alpha=0.8)
        plt.title("State Value Estimates V(s)")
        plt.xlabel("Training Steps")
        plt.ylabel("V(s)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/pendulum_training_stats.png", dpi=150)
        plt.close()
        logger.info("Saved training stats to results/pendulum_training_stats.png")

        df = pd.DataFrame({"loss": agent.losses, "value": agent.vs})
        df.to_csv("results/pendulum_stats.csv", index=False)

    # Evaluate across 10 episodes and plot a trajectory
    eval_rewards = []
    states, actions, rewards = [], [], []
    for ep in range(10):
        obs, _ = env.reset(seed=seed + 100 + ep)
        ep_ret = 0.0
        for _ in range(200):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            if ep == 0:
                rewards.append(reward)
                states.append(obs)
                actions.append(action)
            if terminated or truncated:
                break
        eval_rewards.append(ep_ret)

    logger.info(
        f"Evaluation Reward (10 episodes): {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}"
    )

    states = np.array(states)
    actions = np.array(actions)

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(rewards, color="#d62728")
    plt.title(f"Evaluation Rollout on Pendulum-v1 (Total Return: {eval_rewards[0]:.1f})")
    plt.ylabel("Step Reward")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(states[:, 0], label="cos(theta)")
    plt.plot(states[:, 1], label="sin(theta)")
    plt.plot(states[:, 2] / 8.0, label="theta_dot / 8", linestyle="--")
    plt.legend(loc="upper right")
    plt.title("Normalized Pendulum States")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(actions, color="#9467bd")
    plt.title("Control Action (Torque in [-2, 2] Nm)")
    plt.ylabel("Torque")
    plt.xlabel("Step")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/pendulum_eval_plot.png", dpi=150)
    plt.close()
    logger.info("Saved eval plot to results/pendulum_eval_plot.png")

    env.close()


if __name__ == "__main__":
    main()
