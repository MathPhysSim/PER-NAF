"""CERN Linac4 (L4T) 16-DOF beam-steering simulation environment.

Implements the 16-corrector / 16-BPM trajectory steering problem using the
measured Linac4 transfer-line response matrix (`InfrastructuralData/linac4_rm.pcl`),
as studied in:
  V. Kain, S. Hirlaender et al., "Sample-efficient reinforcement learning for
  CERN accelerator control", Phys. Rev. Accel. Beams 23, 124801 (2020), Sec. IV.
"""

from pathlib import Path
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


def load_linac4_response_matrix(rm_path=None, transpose=False) -> np.ndarray:
    if rm_path is None:
        base_dir = Path(__file__).resolve().parent
        candidates = [
            Path("InfrastructuralData/linac4_rm.pcl"),
            base_dir / "InfrastructuralData" / "linac4_rm.pcl",
            Path("linac4_rm.pcl"),
        ]
        for c in candidates:
            if c.exists():
                rm_path = c
                break
    if rm_path is None or not Path(rm_path).exists():
        raise FileNotFoundError("Could not locate linac4_rm.pcl in InfrastructuralData/")

    df = pd.read_pickle(rm_path)
    if transpose:
        # Legacy orientation from simple_environment_linac4.py
        return df.T.iloc[:16, 1:17].values.astype(np.float64)
    # Physical orientation: rows = 16 Horizontal BPMs, cols = 16 Horizontal Correctors
    return df.iloc[1:17, :16].values.astype(np.float64)


class Linac4Env(gym.Env):
    """Linac4 16-DOF beam-steering environment (Gymnasium API).

    Parameters
    ----------
    dof : int, optional
        Number of active correctors/BPMs (up to 16, default 16).
    incremental : bool, default True
        If True (physical MDP in PRAB 2020), actions represent incremental
        dipole trims (`settings_t = settings_{t-1} + action * action_scale`)
        steering a displaced beam toward the reference orbit.
        If False, matches the legacy `simple_environment_linac4.py` formulation.
    action_scale : float, default 0.6
        Scaling factor applied to normalized actions in [-1, 1].
    init_scale : float, default 0.5
        Half-width of initial corrector displacement on reset.
    threshold : float, default -0.20
        Target negative RMS threshold for successful episode termination.
    max_time : int, default 25
        Maximum steps per episode before time-limit truncation.
    noise_sigma : float, default 0.005
        Shot-to-shot measurement/response noise standard deviation.
    nonlinearity : float, default 0.02
        Magnitude of the quadratic non-linear perturbation component.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__version__ = "0.0.2"
        self.__name__ = f"Linac4Env - Version {self.__version__}"

        transpose = bool(kwargs.get("transpose", False))
        A = load_linac4_response_matrix(kwargs.get("rm_path", None), transpose=transpose)
        dof = kwargs.get("dof", None)
        if dof is not None:
            dof = int(dof)
            A = A[:dof, :dof]

        self.response_matrix = A
        self.obs_dimension = A.shape[0]
        self.act_dimension = A.shape[1]

        self.incremental = bool(kwargs.get("incremental", True))
        self.action_scale = float(kwargs.get("action_scale", 0.6))
        self.init_scale = float(kwargs.get("init_scale", 0.6))
        self.threshold = float(kwargs.get("threshold", -0.25))
        self.MAX_TIME = int(kwargs.get("max_time", 25))
        self.noise_sigma = float(kwargs.get("noise_sigma", 0.005))
        self.nonlinearity = float(kwargs.get("nonlinearity", 0.02))

        self.MAX_POS = 1.0
        act_high = np.ones(self.act_dimension, dtype=np.float32) * self.MAX_POS
        obs_high = np.ones(self.obs_dimension, dtype=np.float32) * self.MAX_POS
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.rng = np.random.default_rng(kwargs.get("seed", 123))

        self.reference_trajectory = (
            np.zeros(self.obs_dimension, dtype=np.float64)
            if self.incremental
            else np.ones(self.obs_dimension, dtype=np.float64) * 0.25
        )

        self.settings = np.zeros(self.act_dimension, dtype=np.float64)
        self.is_finalized = False
        self.curr_step = 0
        self.curr_episode = -1
        self.TOTAL_COUNTER = 0
        self.action_episode_memory = []
        self.rewards = []
        self.initial_conditions = []
        self.success = 0

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def _compute_orbit(self, settings: np.ndarray) -> tuple[np.ndarray, float]:
        self.TOTAL_COUNTER += 1
        if self.nonlinearity > 0.0:
            eff_settings = settings + self.nonlinearity * (settings * np.abs(settings))
        else:
            eff_settings = settings

        state = self.response_matrix @ eff_settings - self.reference_trajectory
        if self.noise_sigma > 0.0:
            state = state + self.noise_sigma * self.rng.standard_normal(self.obs_dimension)

        reward = float(-np.sqrt(np.mean(np.square(state))))
        return state.astype(np.float32), reward

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, -self.MAX_POS, self.MAX_POS)

        self.curr_step += 1
        if self.incremental:
            self.settings = self.settings + action * self.action_scale
        else:
            self.settings = action * self.action_scale

        state, reward = self._compute_orbit(self.settings)
        self.action_episode_memory[self.curr_episode].append(action)
        self.rewards[self.curr_episode].append(reward)

        terminated = False
        truncated = False

        if reward > self.threshold:
            terminated = True
            self.success = 1
        elif np.any(np.abs(state) > 15.0 * abs(self.threshold)):
            truncated = True

        if self.curr_step >= self.MAX_TIME and not terminated:
            truncated = True

        self.is_finalized = terminated or truncated
        info = {"is_success": bool(self.success), "rms": -reward, "steps": self.curr_step}
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.curr_episode += 1
        self.curr_step = 0
        self.is_finalized = False
        self.success = 0
        self.action_episode_memory.append([])
        self.rewards.append([])

        bad_init = True
        init_state = None
        init_reward = 0.0
        while bad_init:
            self.settings = self.rng.uniform(
                -self.init_scale, self.init_scale, size=self.act_dimension
            )
            init_state, init_reward = self._compute_orbit(self.settings)
            bad_init = (
                (init_reward > self.threshold)
                or np.any(np.abs(init_state) > 10.0 * abs(self.threshold))
            )

        self.initial_conditions.append(init_state)
        info = {"rms": -init_reward}
        return init_state, info


# Alias for backward compatibility
simpleEnv = Linac4Env


if __name__ == "__main__":
    env = Linac4Env()
    obs, info = env.reset(seed=42)
    print("Linac4Env initial RMS:", info["rms"], "threshold:", -env.threshold)
    for step_idx in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        print(f"Step {step_idx+1}: reward={r:.4f}, term={term}, trunc={trunc}")
