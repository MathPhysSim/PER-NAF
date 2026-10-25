"""
High-performance JIT-compiled (Numba LLVM + NumPy fallback) Replay Buffers
(Uniform and Prioritized Experience Replay) for continuous control and NAF / PER-NAF agents.

Uses:
1. Pre-allocated contiguous float32 NumPy ring buffers for zero-copy PyTorch tensor conversion.
2. Flat contiguous float64 binary SumTree & MinTree compiled with Numba @njit (fastmath=True)
   for microsecond-level O(B log C) stratified sampling and priority updates.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@njit(cache=True, fastmath=True)
def _numba_tree_set(
    sum_tree: np.ndarray,
    min_tree: np.ndarray,
    tree_capacity: int,
    idx: int,
    p_alpha: float,
) -> None:
    """Update a single leaf at data index `idx` and propagate up to root in LLVM machine code."""
    tree_idx = idx + tree_capacity
    sum_tree[tree_idx] = p_alpha
    min_tree[tree_idx] = p_alpha
    tree_idx >>= 1
    while tree_idx >= 1:
        left = tree_idx << 1
        right = left + 1
        sum_tree[tree_idx] = sum_tree[left] + sum_tree[right]
        l_min = min_tree[left]
        r_min = min_tree[right]
        min_tree[tree_idx] = l_min if l_min < r_min else r_min
        tree_idx >>= 1


@njit(cache=True, fastmath=True)
def _numba_batch_sample(
    sum_tree: np.ndarray,
    min_tree: np.ndarray,
    tree_capacity: int,
    size: int,
    batch_size: int,
    beta: float,
    rand_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified proportional sampling + importance-sampling weights in a single LLVM loop."""
    idxs = np.empty(batch_size, dtype=np.int64)
    weights = np.empty(batch_size, dtype=np.float32)

    p_total = sum_tree[1]
    if p_total <= 0.0 or not np.isfinite(p_total):
        for i in range(batch_size):
            idxs[i] = int(rand_u[i] * size) % size
            weights[i] = 1.0
        return idxs, weights

    seg = p_total / batch_size
    p_min = min_tree[1]
    if p_min < 1e-12 or not np.isfinite(p_min):
        p_min = 1e-12

    for i in range(batch_size):
        target = (i + rand_u[i]) * seg
        node = 1
        while node < tree_capacity:
            left = node << 1
            left_val = sum_tree[left]
            if target <= left_val:
                node = left
            else:
                target -= left_val
                node = left + 1
        data_idx = node - tree_capacity
        if data_idx >= size:
            data_idx = size - 1
        elif data_idx < 0:
            data_idx = 0
        idxs[i] = data_idx

        p_sample = sum_tree[data_idx + tree_capacity]
        if p_sample < 1e-12:
            p_sample = 1e-12
        weights[i] = np.float32(p_sample ** (-beta))

    w_max = np.float32(0.0)
    for i in range(batch_size):
        if weights[i] > w_max:
            w_max = weights[i]
    if w_max > 1e-12:
        for i in range(batch_size):
            weights[i] /= w_max

    return idxs, weights


@njit(cache=True, fastmath=True)
def _numba_batch_update(
    sum_tree: np.ndarray,
    min_tree: np.ndarray,
    tree_capacity: int,
    idx_arr: np.ndarray,
    prio_arr: np.ndarray,
    alpha: float,
    eps: float,
    max_clip: float,
    current_max_prio: float,
) -> float:
    """Batch update of priorities in LLVM machine code."""
    n = idx_arr.shape[0]
    max_p = current_max_prio
    for i in range(n):
        p = prio_arr[i]
        if p < eps:
            p = eps
        elif max_clip > 0.0 and p > max_clip:
            p = max_clip
        if p > max_p:
            max_p = p
        p_alpha = p ** alpha
        _numba_tree_set(sum_tree, min_tree, tree_capacity, int(idx_arr[i]), p_alpha)
    return max_p


class ReplayBuffer:
    """
    Fast contiguous NumPy FIFO experience replay buffer.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.max_size = int(size)
        self.ptr = 0
        self.size = 0

        self.obs1_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((self.max_size, self.act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: Union[bool, float],
    ) -> None:
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = float(rew)
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def sample_batch(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        if self.size < batch_size:
            idxs = np.arange(self.size, dtype=np.int64)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )


class ReplayBufferPER(ReplayBuffer):
    """
    Numba-JIT accelerated Prioritized Experience Replay (PER) buffer using contiguous
    NumPy arrays and a flat binary SumTree + MinTree.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        prio_info: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, size=size)
        prio_info = dict(prio_info) if prio_info else {}
        prio_info.update(kwargs)
        self.alpha = float(prio_info.get("alpha", 0.6))
        self.beta_start = float(prio_info.get("beta", prio_info.get("beta_start", 0.4)))
        self.beta = self.beta_start
        self.eps = float(prio_info.get("eps", 1e-6))
        max_clip = prio_info.get("max_priority_clip", 100.0)
        self.max_priority_clip = float(max_clip) if max_clip is not None else -1.0

        assert self.alpha >= 0.0, "PER alpha must be non-negative"

        self.tree_capacity = 1 << max(0, (self.max_size - 1).bit_length())
        self.tree_depth = int(np.log2(self.tree_capacity))

        self._sum_tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)
        self._min_tree = np.full(2 * self.tree_capacity, np.inf, dtype=np.float64)
        self._max_priority = 1.0
        self._calibrated_scale = False

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: Union[bool, float],
    ) -> None:
        idx = self.ptr
        super().store(obs, act, rew, next_obs, done)
        p_alpha = self._max_priority ** self.alpha
        _numba_tree_set(self._sum_tree, self._min_tree, self.tree_capacity, idx, p_alpha)

    def sample_batch(
        self, batch_size: int = 32, beta: Optional[float] = None
    ) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        if beta is None:
            beta = self.beta
        else:
            beta = float(beta)
        beta = max(beta, 1e-8)

        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBufferPER.")

        if self.size < batch_size:
            idxs = np.arange(self.size, dtype=np.int64)
            weights = np.ones(self.size, dtype=np.float32)
        else:
            rand_u = np.random.random(batch_size)
            idxs, weights = _numba_batch_sample(
                self._sum_tree,
                self._min_tree,
                self.tree_capacity,
                self.size,
                batch_size,
                beta,
                rand_u,
            )

        batch = dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch, [weights, idxs]

    def update_priorities(
        self, idxes: Union[np.ndarray, Sequence[int]], priorities: Union[np.ndarray, Sequence[float]]
    ) -> None:
        idx_arr = np.asarray(idxes, dtype=np.int64)
        prio_arr = np.asarray(priorities, dtype=np.float64)
        if not self._calibrated_scale and self.size > 0:
            # Calibrate initial placeholder priorities (1.0) to empirical TD error scale
            emp_max = float(np.max(np.clip(prio_arr, self.eps, None)))
            self._max_priority = emp_max
            p_alpha_init = emp_max ** self.alpha
            for k in range(self.size):
                _numba_tree_set(self._sum_tree, self._min_tree, self.tree_capacity, k, p_alpha_init)
            self._calibrated_scale = True

        self._max_priority = _numba_batch_update(
            self._sum_tree,
            self._min_tree,
            self.tree_capacity,
            idx_arr,
            prio_arr,
            self.alpha,
            self.eps,
            self.max_priority_clip,
            self._max_priority * 0.995,
        )


PrioritizedReplayBuffer = ReplayBufferPER
PriortizedReplayBuffer = ReplayBufferPER

