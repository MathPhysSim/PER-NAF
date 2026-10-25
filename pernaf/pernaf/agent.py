"""
Normalized Advantage Function (NAF / NAF2 / PER-NAF / PER-NAF2) Agent.

Supports two high-performance execution backends:
- `backend="jax"` (default when JAX + Optax are installed): XLA-compiled `@jax.jit` network
  forward/backward + Optax Adam + Numba `@njit` LLVM SumTree (~10x faster end-to-end).
- `backend="torch"`: Vectorized PyTorch 2.x `QNetwork` + `_foreach_lerp_` + Numba `@njit` SumTree.
Both backends share identical initial weights and sync seamlessly to `self.q_main` / `self.q2_main`.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .networks import QNetwork
from .buffers import ReplayBuffer, ReplayBufferPER

try:
    from .jax_naf import JaxNAFUpdater
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class NAF:
    """
    Normalized Advantage Function (NAF) Agent.
    Supports Clipped Double Q-Learning (NAF2) and Prioritized Experience Replay (PER).
    """

    def __init__(
        self,
        env,
        learning_rate: float = 1e-3,
        buffer_size: int = 1_000_000,
        batch_size: int = 100,
        discount: float = 0.99,
        polyak: float = 0.995,
        tau: Optional[float] = None,
        max_steps: int = 200,
        update_repeat: int = 5,
        warmup_steps: int = 0,
        prio_info: Optional[Dict] = None,
        noise_info: Optional[Dict] = None,
        directory: Optional[str] = None,
        device: str = "cpu",
        backend: str = "auto",
        double_q: bool = False,
        double_q_mode: str = "min_v",
        target_smoothing_sigma: float = 0.0,
        target_smoothing_clip: float = 0.05,
        loss_type: str = "auto",
        huber_delta: float = 1.0,
        grad_clip_norm: Optional[float] = 10.0,
        normalize_obs: Union[bool, str] = "auto",
        rescale_actions: bool = True,
        pretune: Optional[List] = None,
        verbose: bool = True,
        **nafnet_kwargs,
    ):
        self.env = env
        self.learning_rate = float(learning_rate)
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.discount = float(discount)
        if tau is not None:
            self.polyak = 1.0 - float(tau)
        else:
            self.polyak = float(polyak)
        self.max_steps = int(max_steps)
        self.update_repeat = int(update_repeat)
        self.warmup_steps = int(warmup_steps)
        self.prio_info = dict(prio_info) if prio_info else {}
        self.noise_info = dict(noise_info) if noise_info else {}
        self.directory = directory
        self.nafnet_kwargs = nafnet_kwargs
        self.double_q = bool(double_q)
        self.double_q_mode = str(double_q_mode)
        self.target_smoothing_sigma = float(target_smoothing_sigma)
        self.target_smoothing_clip = float(target_smoothing_clip)
        self.grad_clip_norm = grad_clip_norm
        self.pretune = pretune
        self.verbose = bool(verbose)

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        # PER setup
        self.per_flag = bool(self.prio_info) and float(self.prio_info.get("alpha", 0.0)) > 0.0
        if loss_type == "auto":
            self.loss_type = "huber" if self.per_flag else "mse"
        else:
            self.loss_type = str(loss_type).lower()
        self.huber_delta = float(huber_delta)

        # Logging / Statistics
        self.losses: List[float] = []
        self.vs: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.counter = 0

        # Exploration noise schedule
        if "noise_function" in self.noise_info:
            self.noise_function: Callable[[int], float] = self.noise_info["noise_function"]
        else:
            self.noise_function = lambda nr: 1.0 / (nr + 1.0)

        # Dimensions & Space Scaling
        self.action_size = int(env.action_space.shape[0])
        self.obs_dim = int(env.observation_space.shape[0])

        self.rescale_actions = bool(rescale_actions)
        act_low = np.asarray(env.action_space.low, dtype=np.float32)
        act_high = np.asarray(env.action_space.high, dtype=np.float32)
        if self.rescale_actions and np.all(np.isfinite(act_low)) and np.all(np.isfinite(act_high)):
            self.act_scale = 0.5 * (act_high - act_low)
            self.act_bias = 0.5 * (act_high + act_low)
        else:
            self.act_scale = np.ones(self.action_size, dtype=np.float32)
            self.act_bias = np.zeros(self.action_size, dtype=np.float32)

        obs_low = np.asarray(env.observation_space.low, dtype=np.float32)
        obs_high = np.asarray(env.observation_space.high, dtype=np.float32)
        finite_obs = np.all(np.isfinite(obs_low)) and np.all(np.isfinite(obs_high)) and np.all((obs_high - obs_low) > 0)
        if normalize_obs == "auto":
            is_unit = finite_obs and np.allclose(obs_low, -1.0) and np.allclose(obs_high, 1.0)
            self.normalize_obs = bool(finite_obs and not is_unit)
        else:
            self.normalize_obs = bool(normalize_obs and finite_obs)

        if self.normalize_obs:
            self.obs_scale = 0.5 * (obs_high - obs_low)
            self.obs_bias = 0.5 * (obs_high + obs_low)
        else:
            self.obs_scale = np.ones(self.obs_dim, dtype=np.float32)
            self.obs_bias = np.zeros(self.obs_dim, dtype=np.float32)

        # Initialize Replay Buffer
        if not self.per_flag:
            self.replay_buffer = ReplayBuffer(
                obs_dim=self.obs_dim, act_dim=self.action_size, size=self.buffer_size
            )
        else:
            self.replay_buffer = ReplayBufferPER(
                obs_dim=self.obs_dim,
                act_dim=self.action_size,
                size=self.buffer_size,
                prio_info=self.prio_info,
            )

        # Beta schedule for PER importance-sampling weights
        self.beta_start = float(self.prio_info.get("beta", self.prio_info.get("beta_start", 0.4)))
        self.beta_end = float(self.prio_info.get("beta_end", 1.0))
        if "decay_function" in self.prio_info:
            self.beta_schedule = self.prio_info["decay_function"]
        elif "beta_decay" in self.prio_info:
            self.beta_schedule = self.prio_info["beta_decay"]
        else:
            self.beta_schedule = None

        # Initialize PyTorch Q-Networks
        self.q_main = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
        self.q_target = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q_main.state_dict())
        self.q_target.eval()

        self._main_params = list(self.q_main.parameters())
        self._target_params = list(self.q_target.parameters())

        if self.double_q:
            self.q2_main = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
            self.q2_target = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
            self.q2_target.load_state_dict(self.q2_main.state_dict())
            self.q2_target.eval()

            self._main_params = self._main_params + list(self.q2_main.parameters())
            self._target_params = self._target_params + list(self.q2_target.parameters())
        else:
            self.q2_main = None
            self.q2_target = None

        self.optimizer = optim.Adam(self._main_params, lr=self.learning_rate)

        # Select Backend ("jax" or "torch")
        can_use_jax = (
            HAS_JAX
            and not self.q_main.layer_norm
            and self.target_smoothing_sigma == 0.0
        )
        if backend == "auto":
            self.backend = "jax" if can_use_jax else "torch"
        elif backend == "jax" and can_use_jax:
            self.backend = "jax"
        else:
            self.backend = "torch"

        self._jax_updater: Optional[JaxNAFUpdater] = None
        if self.backend == "jax":
            self._jax_updater = JaxNAFUpdater(
                obs_dim=self.obs_dim,
                act_dim=self.action_size,
                hidden_sizes=self.q_main.hidden_sizes,
                learning_rate=self.learning_rate,
                discount=self.discount,
                polyak=self.polyak,
                double_q=self.double_q,
                double_q_mode=self.double_q_mode,
                loss_type=self.loss_type,
                huber_delta=self.huber_delta,
                grad_clip_norm=self.grad_clip_norm,
                q_main_torch=self.q_main,
                q2_main_torch=self.q2_main,
                l_scale=self.q_main.l_scale,
                diag_clamp=self.q_main.diag_clamp,
            )

        if self.verbose:
            print(
                f"NAF Agent (Backend={self.backend.upper()}, DoubleQ={self.double_q}, "
                f"PER={self.per_flag}, Loss={self.loss_type}) initialized."
            )

        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)

    def _scale_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.normalize_obs:
            return (np.asarray(obs, dtype=np.float32) - self.obs_bias) / self.obs_scale
        return np.asarray(obs, dtype=np.float32)

    def _to_env_action(self, norm_action: np.ndarray) -> np.ndarray:
        if self.rescale_actions:
            return norm_action * self.act_scale + self.act_bias
        return norm_action

    def _to_norm_action(self, env_action: np.ndarray) -> np.ndarray:
        if self.rescale_actions:
            return (env_action - self.act_bias) / self.act_scale
        return env_action

    def _predict_norm(self, norm_obs: np.ndarray) -> np.ndarray:
        """Fast internal action prediction in normalized [-1, 1] space."""
        if self._jax_updater is not None:
            return self._jax_updater.predict_norm(norm_obs)
        self.q_main.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu = self.q_main.forward_mu(obs_tensor)
            action_norm = mu.cpu().numpy()[0]
        self.q_main.train()
        return action_norm

    def predict(
        self, observation: np.ndarray, state=None, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[object]]:
        """Predict action in environment coordinates [action_space.low, action_space.high]."""
        norm_obs = self._scale_obs(observation)
        action_norm = self._predict_norm(norm_obs)
        if not deterministic:
            noise_scale = self.noise_function(len(self.episode_rewards) + 1)
            action_norm = np.clip(action_norm + noise_scale * np.random.randn(self.action_size), -1.0, 1.0)
        return self._to_env_action(action_norm), state

    def _step_env(self, env_action: np.ndarray):
        step_out = self.env.step(env_action)
        if len(step_out) == 5:
            o2, r, terminated, truncated, info = step_out
            return o2, float(r), bool(terminated), bool(truncated), info
        o2, r, done, info = step_out
        return o2, float(r), bool(done), False, info

    def _reset_env(self):
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
            return reset_out[0], reset_out[1]
        return reset_out, {}

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 100,
        progress_bar: bool = True,
    ):
        total_timesteps = int(total_timesteps)
        steps = 0
        episodes = 0

        if self.pretune:
            for item in self.pretune:
                o_p, a_p, r_p, o2_p, d_p = item[:5]
                self.replay_buffer.store(
                    self._scale_obs(o_p),
                    self._to_norm_action(np.asarray(a_p, dtype=np.float32)),
                    float(r_p),
                    self._scale_obs(o2_p),
                    float(d_p),
                )

        o, _ = self._reset_env()
        o_norm = self._scale_obs(o)

        pbar = tqdm(total=total_timesteps, disable=not (progress_bar and self.verbose))

        while steps < total_timesteps:
            episodes += 1
            noise_scale = float(self.noise_function(episodes))
            episode_reward = 0.0
            ep_steps = 0

            done = False
            while not done and steps < total_timesteps:
                ep_steps += 1
                if steps < self.warmup_steps:
                    action_norm = np.random.uniform(-1.0, 1.0, size=self.action_size).astype(np.float32)
                else:
                    action_mean = self._predict_norm(o_norm)
                    action_norm = action_mean + noise_scale * np.random.randn(self.action_size).astype(np.float32)
                    action_norm = np.clip(action_norm, -1.0, 1.0)

                env_action = self._to_env_action(action_norm)
                o2, r, terminated, truncated, _ = self._step_env(env_action)

                if self.max_steps > 0 and ep_steps >= self.max_steps and not terminated:
                    truncated = True

                done = terminated or truncated
                terminal_flag = 1.0 if (terminated and not truncated) else 0.0

                o2_norm = self._scale_obs(o2)
                episode_reward += r

                self.replay_buffer.store(o_norm, action_norm, r, o2_norm, terminal_flag)

                o_norm = o2_norm
                steps += 1
                pbar.update(1)

                min_buffer_ready = max(self.batch_size, self.warmup_steps)
                if self.replay_buffer.size >= min_buffer_ready:
                    if self.per_flag:
                        if self.beta_schedule is not None:
                            current_beta = float(self.beta_schedule(episodes))
                        else:
                            frac = min(1.0, steps / max(1, total_timesteps))
                            current_beta = self.beta_start + frac * (self.beta_end - self.beta_start)
                        self.replay_buffer.beta = max(1e-6, current_beta)

                    for _ in range(self.update_repeat):
                        self.update_q(episodes)

                if callback is not None:
                    callback(self, steps, episodes)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(ep_steps)
            if done and steps < total_timesteps:
                o, _ = self._reset_env()
                o_norm = self._scale_obs(o)

        pbar.close()
        if self._jax_updater is not None:
            self._jax_updater.sync_to_torch(self.q_main, self.q_target, self.q2_main, self.q2_target)
        return self

    def run(self, is_train: bool = True):
        max_episodes = int(self.nafnet_kwargs.get("max_episodes", 100))
        if is_train:
            return self.learn(total_timesteps=max_episodes * max(1, self.max_steps))
        for _ in range(max_episodes):
            o, _ = self._reset_env()
            for _ in range(self.max_steps):
                a, _ = self.predict(o, deterministic=True)
                o, _, terminated, truncated, _ = self._step_env(a)
                if terminated or truncated:
                    break
        return self

    def _elementwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, reduction="none", beta=self.huber_delta)
        return F.mse_loss(pred, target, reduction="none")

    def update_q(self, episode_idx: int = 0) -> None:
        self.counter += 1

        if self.per_flag:
            batch, priority_info = self.replay_buffer.sample_batch(self.batch_size)
            weights, idxs = priority_info[0], priority_info[1]
        else:
            batch = self.replay_buffer.sample_batch(self.batch_size)
            weights, idxs = None, None

        # Fast path: XLA-compiled JAX backend
        if self._jax_updater is not None:
            loss_val, td_err, v_val = self._jax_updater.update(batch, weights)
            if self.per_flag:
                self.replay_buffer.update_priorities(idxes=idxs, priorities=td_err + 1e-6)
            self.losses.append(loss_val)
            self.vs.append(v_val)
            return

        # PyTorch backend
        o = torch.as_tensor(batch["obs1"], dtype=torch.float32, device=self.device)
        o2 = torch.as_tensor(batch["obs2"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["acts"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch["rews"], dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            if self.double_q:
                if self.double_q_mode == "clipped_q" or self.target_smoothing_sigma > 0.0:
                    mu_next = self.q_target.forward_mu(o2)
                    if self.target_smoothing_sigma > 0.0:
                        noise = torch.clamp(
                            torch.randn_like(mu_next) * self.target_smoothing_sigma,
                            -self.target_smoothing_clip,
                            self.target_smoothing_clip,
                        )
                        a_next = torch.clamp(mu_next + noise, -1.0, 1.0)
                    else:
                        a_next = mu_next
                    q_next_1, _, _ = self.q_target.forward_q(o2, a_next)
                    q_next_2, _, _ = self.q2_target.forward_q(o2, a_next)
                    V_next = torch.min(q_next_1, q_next_2)
                else:
                    V_next_1 = self.q_target.forward_v(o2)
                    V_next_2 = self.q2_target.forward_v(o2)
                    V_next = torch.min(V_next_1, V_next_2)
            else:
                V_next = self.q_target.forward_v(o2)

            target_q = r + (1.0 - d) * self.discount * V_next

        if self.double_q:
            Q1, V1, _ = self.q_main.forward_q(o, a)
            Q2, _, _ = self.q2_main.forward_q(o, a)

            if self.per_flag:
                w_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
                loss1 = (w_t * self._elementwise_loss(Q1, target_q)).mean()
                loss2 = (w_t * self._elementwise_loss(Q2, target_q)).mean()
            else:
                loss1 = self._elementwise_loss(Q1, target_q).mean()
                loss2 = self._elementwise_loss(Q2, target_q).mean()
            loss = loss1 + loss2
        else:
            Q, V1, _ = self.q_main.forward_q(o, a)
            if self.per_flag:
                w_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
                loss = (w_t * self._elementwise_loss(Q, target_q)).mean()
            else:
                loss = self._elementwise_loss(Q, target_q).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._main_params, self.grad_clip_norm)
        self.optimizer.step()

        if self.per_flag:
            with torch.no_grad():
                if self.double_q:
                    td_error = 0.5 * (torch.abs(Q1 - target_q) + torch.abs(Q2 - target_q))
                else:
                    td_error = torch.abs(Q - target_q)
                new_priorities = td_error.squeeze(1).cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(idxes=idxs, priorities=new_priorities)

        with torch.no_grad():
            torch._foreach_lerp_(self._target_params, self._main_params, 1.0 - self.polyak)

        self.losses.append(float(loss.item()))
        self.vs.append(float(V1.mean().item()))

    def save(self, path: str) -> None:
        if self._jax_updater is not None:
            self._jax_updater.sync_to_torch(self.q_main, self.q_target, self.q2_main, self.q2_target)
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        torch.save(self.q_main.state_dict(), path + ".pth")
        if self.double_q and self.q2_main is not None:
            torch.save(self.q2_main.state_dict(), path + "_q2.pth")
        if self.verbose:
            print(f"Model saved to {path}.pth")

    def load(self, path: str) -> None:
        self.q_main.load_state_dict(torch.load(path + ".pth", map_location=self.device, weights_only=True))
        self.q_target.load_state_dict(self.q_main.state_dict())
        if self.double_q and self.q2_main is not None and os.path.exists(path + "_q2.pth"):
            self.q2_main.load_state_dict(
                torch.load(path + "_q2.pth", map_location=self.device, weights_only=True)
            )
            self.q2_target.load_state_dict(self.q2_main.state_dict())
        if self._jax_updater is not None:
            from .jax_naf import init_naf_params_from_torch
            import jax
            self._jax_updater.p1 = init_naf_params_from_torch(self.q_main)
            self._jax_updater.tp1 = jax.tree_util.tree_map(lambda x: x.copy(), self._jax_updater.p1)
            if self.double_q and self.q2_main is not None:
                self._jax_updater.p2 = init_naf_params_from_torch(self.q2_main)
                self._jax_updater.tp2 = jax.tree_util.tree_map(lambda x: x.copy(), self._jax_updater.p2)
        if self.verbose:
            print(f"Model loaded from {path}.pth")
