import logging
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pernaf.pernaf.agent import NAF


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_experiment(name, config, seed=123, steps=20000):
    logging.info(f"--- Starting {name} (seed={seed}) ---")
    set_seed(seed)

    env = gym.make("Pendulum-v1")
    cfg = dict(config)
    cfg["seed"] = seed
    cfg["verbose"] = False
    agent = NAF(env, **cfg)
    agent.learn(total_timesteps=steps, progress_bar=False)
    env.close()
    return np.asarray(agent.episode_rewards, dtype=np.float64)


def moving_average(data, window_size=10):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values


def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs("results", exist_ok=True)

    steps = 20000
    seeds = [101, 202, 303]
    tuned_noise = dict(noise_function=lambda nr: max(0.05, 1.0 / (1.0 + nr * 0.08)))

    base_cfg = dict(
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=50000,
        discount=0.98,
        polyak=0.995,
        update_repeat=2,
        warmup_steps=400,
        net_arch=[64, 64],
        noise_info=tuned_noise,
    )

    variants = {
        "NAF (Single-Q, Uniform)": dict(**base_cfg, prio_info={}, double_q=False),
        "NAF2 (Double-Q, Uniform)": dict(**base_cfg, prio_info={}, double_q=True),
        "PER-NAF (Single-Q + PER)": dict(**base_cfg, prio_info=dict(alpha=0.6, beta=0.4), double_q=False),
        "PER-NAF2 (Double-Q + PER)": dict(**base_cfg, prio_info=dict(alpha=0.6, beta=0.4), double_q=True),
    }

    colors = {
        "NAF (Single-Q, Uniform)": "#1f77b4",
        "NAF2 (Double-Q, Uniform)": "#ff7f0e",
        "PER-NAF (Single-Q + PER)": "#2ca02c",
        "PER-NAF2 (Double-Q + PER)": "#d62728",
    }

    plt.figure(figsize=(12, 6))
    summary_rows = []

    for label, cfg in variants.items():
        all_runs = [run_experiment(label, cfg, seed=s, steps=steps) for s in seeds]
        min_len = min(len(r) for r in all_runs)
        mat = np.stack([moving_average(r[:min_len], window_size=10) for r in all_runs], axis=0)
        raw_mat = np.stack([r[:min_len] for r in all_runs], axis=0)

        mean_curve = mat.mean(axis=0)
        sem_curve = mat.std(axis=0) / np.sqrt(len(seeds))
        x = np.arange(1, min_len + 1)

        plt.plot(x, mean_curve, label=label, color=colors[label], linewidth=2.2)
        plt.fill_between(
            x, mean_curve - sem_curve, mean_curve + sem_curve, color=colors[label], alpha=0.18
        )

        final_20 = raw_mat[:, -20:].mean(axis=1)
        auc = raw_mat.mean(axis=1)
        summary_rows.append(
            {
                "variant": label,
                "final_20_mean": float(final_20.mean()),
                "final_20_std": float(final_20.std()),
                "auc_mean": float(auc.mean()),
                "auc_std": float(auc.std()),
            }
        )

    plt.title(f"2x2 Factorial Ablation on Pendulum-v1 ({steps} steps, {len(seeds)} seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return (10-ep moving average)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/naf_vs_naf2_rewards.png", dpi=150)
    plt.close()

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/pendulum_factorial_summary.csv", index=False)
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
