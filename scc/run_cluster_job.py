"""SCC Cluster Experiment Runner for Large-Scale PER-NAF / NAF2 Factorial Ablations.

Supports SLURM job arrays (`--task-id $SLURM_ARRAY_TASK_ID`) where each task runs
one seed across all scenarios and hyperparameter combinations, writing atomic
JSON/CSV checkpoints to `scc/outputs/` immediately after each scenario finishes.

Usage on SCC Cluster:
  # Run a single array task (e.g. task 0 -> seed 1000):
  python3 scc/run_cluster_job.py --task-id 0 --out-dir scc/outputs

  # Aggregate all finished tasks into summary CSVs and ANOVA tables:
  python3 scc/run_cluster_job.py --aggregate --out-dir scc/outputs
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy import stats

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pernaf.pernaf.agent import NAF
from simulated_environment import AwakeElectronEnv
from linac4_environment import Linac4Env


VARIANTS = {
    "NAF": dict(double_q=False, prio_info={}),
    "NAF2": dict(double_q=True, prio_info={}),
    "PER-NAF_a04": dict(double_q=False, prio_info=dict(alpha=0.4, beta=0.5)),
    "PER-NAF_a06": dict(double_q=False, prio_info=dict(alpha=0.6, beta=0.5)),
    "PER-NAF_a08": dict(double_q=False, prio_info=dict(alpha=0.8, beta=0.5)),
    "PER-NAF2_a04": dict(double_q=True, prio_info=dict(alpha=0.4, beta=0.5)),
    "PER-NAF2_a06": dict(double_q=True, prio_info=dict(alpha=0.6, beta=0.5)),
    "PER-NAF2_a08": dict(double_q=True, prio_info=dict(alpha=0.8, beta=0.5)),
}


def shots_to_convergence(ep_lengths, target_steps=2.5, window=8):
    s = pd.Series(ep_lengths).rolling(window=window, min_periods=window).mean().values
    idx = np.where(s <= target_steps)[0]
    ep_idx = int(idx[0] + 1) if len(idx) > 0 else len(ep_lengths)
    return int(np.sum(ep_lengths[:ep_idx])), ep_idx


def evaluate_agent(agent, env_cls, env_kwargs, n_eval=30, seed_base=50000):
    env = env_cls(**env_kwargs)
    steps_list, succ_list, rms_list = [], [], []
    for i in range(n_eval):
        obs, _ = env.reset(seed=seed_base + i)
        for t in range(1, env.MAX_TIME + 1):
            act, _ = agent.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(act)
            if term or trunc:
                steps_list.append(t)
                succ_list.append(1.0 if term else 0.0)
                rms_list.append(float(-rew))
                break
    return float(np.mean(steps_list)), float(np.mean(succ_list)), float(np.mean(rms_list))


def run_seed_suite(seed: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{seed:04d}.json"
    records = []

    scenarios = [
        ("AWAKE_10DOF_Clean", AwakeElectronEnv, dict(scale=3e-4, max_time=40, noise_sigma=0.0, nonlinearity=0.0), 600),
        ("AWAKE_10DOF_HighNoise", AwakeElectronEnv, dict(scale=3e-4, max_time=40, noise_sigma=0.06, nonlinearity=0.0), 600),
        ("AWAKE_10DOF_Nonlinear", AwakeElectronEnv, dict(scale=3e-4, max_time=40, noise_sigma=0.01, nonlinearity=0.25), 600),
        ("LINAC4_16DOF_Nominal", Linac4Env, dict(init_scale=0.6, action_scale=0.6, threshold=-0.25, max_time=25, noise_sigma=0.005, nonlinearity=0.02), 1500),
        ("LINAC4_16DOF_HighNoise", Linac4Env, dict(init_scale=0.6, action_scale=0.6, threshold=-0.25, max_time=25, noise_sigma=0.04, nonlinearity=0.05), 1500),
    ]

    for scen_name, env_cls, env_kw, total_steps in scenarios:
        for vname, vcfg in VARIANTS.items():
            t0 = time.perf_counter()
            env = env_cls(**env_kw)
            is_l4 = env_cls is Linac4Env
            agent = NAF(
                env,
                learning_rate=1e-3,
                batch_size=64 if is_l4 else 32,
                buffer_size=25000,
                discount=0.95 if is_l4 else 0.99,
                polyak=0.995,
                update_repeat=5,
                net_arch=[100, 100] if is_l4 else [64, 64],
                normalize_obs=False,
                noise_info=dict(noise_function=lambda ep: max(0.02, 1.0 / (1.0 + 0.12 * ep))),
                seed=seed,
                verbose=False,
                **vcfg,
            )
            agent.learn(total_timesteps=total_steps, progress_bar=False)
            shots, eps = shots_to_convergence(agent.episode_lengths, target_steps=2.5, window=8)
            ev_steps, ev_succ, ev_rms = evaluate_agent(agent, env_cls, env_kw, n_eval=25)
            dt = time.perf_counter() - t0

            rec = {
                "seed": seed,
                "scenario": scen_name,
                "variant": vname,
                "double_q": vcfg["double_q"],
                "per_alpha": vcfg["prio_info"].get("alpha", 0.0),
                "shots_to_conv": shots,
                "episodes_to_conv": eps,
                "eval_steps": ev_steps,
                "eval_success_rate": ev_succ,
                "eval_final_rms": ev_rms,
                "wall_time_s": round(dt, 3),
            }
            records.append(rec)
            # Atomic checkpoint after each run
            tmp_file = out_dir / f".tmp_seed_{seed:04d}.json"
            with open(tmp_file, "w") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp_file, out_file)

    # Also run Pendulum-v1 for the 4 core variants
    for vname in ["NAF", "NAF2", "PER-NAF_a06", "PER-NAF2_a06"]:
        vcfg = VARIANTS[vname]
        t0 = time.perf_counter()
        env = gym.make("Pendulum-v1")
        agent = NAF(
            env,
            learning_rate=1e-3,
            batch_size=64,
            buffer_size=50000,
            discount=0.98,
            polyak=0.995,
            update_repeat=2,
            warmup_steps=400,
            net_arch=[64, 64],
            noise_info=dict(noise_function=lambda ep: max(0.05, 1.0 / (1.0 + 0.08 * ep))),
            seed=seed,
            verbose=False,
            **vcfg,
        )
        agent.learn(total_timesteps=20000, progress_bar=False)
        rews = np.asarray(agent.episode_rewards, dtype=np.float64)
        env.close()
        rec = {
            "seed": seed,
            "scenario": "Pendulum-v1",
            "variant": vname,
            "double_q": vcfg["double_q"],
            "per_alpha": vcfg["prio_info"].get("alpha", 0.0),
            "shots_to_conv": int(np.mean(rews[15:40])),
            "episodes_to_conv": int(np.mean(rews[-20:])),
            "eval_steps": float(np.mean(rews[-20:])),
            "eval_success_rate": float(np.mean(rews)),
            "eval_final_rms": float(np.max(rews)),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        }
        records.append(rec)
        tmp_file = out_dir / f".tmp_seed_{seed:04d}.json"
        with open(tmp_file, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp_file, out_file)

    print(f"Completed seed {seed}: saved {len(records)} runs to {out_file}")


def aggregate_results(out_dir: Path):
    files = sorted(glob.glob(str(out_dir / "seed_*.json")))
    if not files:
        print(f"No seed_*.json files found in {out_dir}")
        return
    all_rows = []
    for fpath in files:
        with open(fpath, "r") as f:
            all_rows.extend(json.load(f))
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "scc_aggregated_raw.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby(["scenario", "variant"])
        .agg(
            n_seeds=("seed", "count"),
            shots_mean=("shots_to_conv", "mean"),
            shots_std=("shots_to_conv", "std"),
            eval_steps_mean=("eval_steps", "mean"),
            eval_steps_std=("eval_steps", "std"),
            success_mean=("eval_success_rate", "mean"),
        )
        .reset_index()
    )
    sum_path = out_dir / "scc_aggregated_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Aggregated {len(files)} seed files ({len(df)} runs) -> {sum_path}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None, help="SLURM_ARRAY_TASK_ID (0..N-1)")
    parser.add_argument("--seed", type=int, default=None, help="Explicit random seed")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "scc" / "outputs"))
    parser.add_argument("--aggregate", action="store_true", help="Aggregate all completed seed files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.aggregate:
        aggregate_results(out_dir)
        return

    if args.seed is not None:
        seed = args.seed
    elif args.task_id is not None:
        seed = 1000 + int(args.task_id) * 17
    else:
        env_tid = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        seed = 1000 + int(env_tid) * 17 if env_tid is not None else 1000

    run_seed_suite(seed=seed, out_dir=out_dir)


if __name__ == "__main__":
    main()
