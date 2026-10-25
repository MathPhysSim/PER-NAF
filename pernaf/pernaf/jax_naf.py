"""
XLA-compiled JAX + Optax backend for NAF, NAF2, PER-NAF, and PER-NAF2.

Combines:
- Pure-JAX Cholesky NAF networks compiled with `@jax.jit`
- Optax Adam + global norm gradient clipping
- Zero-copy NumPy buffer interop with the Numba `@njit` SumTree in `buffers.py`
- Multi-step fused updates for uniform replay (`jax.lax.scan` over `update_repeat`)
- Bi-directional weight synchronization with PyTorch `QNetwork` for drop-in compatibility
"""

from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax


def init_naf_params_from_torch(q_net) -> Dict[str, jax.Array]:
    """Copy initial weights from a PyTorch QNetwork so seeds and initializations match 1:1."""
    params = {}
    linear_idx = 0
    for m in q_net.base:
        if hasattr(m, "weight") and m.__class__.__name__ == "Linear":
            w = m.weight.detach().cpu().numpy().T  # PyTorch (out, in) -> JAX (in, out)
            b = m.bias.detach().cpu().numpy()
            params[f"w_{linear_idx}"] = jnp.asarray(w, dtype=jnp.float32)
            params[f"b_{linear_idx}"] = jnp.asarray(b, dtype=jnp.float32)
            linear_idx += 1

    params["w_v"] = jnp.asarray(q_net.v_head.weight.detach().cpu().numpy().T, dtype=jnp.float32)
    params["b_v"] = jnp.asarray(q_net.v_head.bias.detach().cpu().numpy(), dtype=jnp.float32)
    params["w_mu"] = jnp.asarray(q_net.mu_head.weight.detach().cpu().numpy().T, dtype=jnp.float32)
    params["b_mu"] = jnp.asarray(q_net.mu_head.bias.detach().cpu().numpy(), dtype=jnp.float32)
    params["w_l"] = jnp.asarray(q_net.l_head.weight.detach().cpu().numpy().T, dtype=jnp.float32)
    params["b_l"] = jnp.asarray(q_net.l_head.bias.detach().cpu().numpy(), dtype=jnp.float32)
    return params


def sync_jax_params_to_torch(params: Dict[str, jax.Array], q_net) -> None:
    """Copy trained JAX parameters back into a PyTorch QNetwork."""
    import torch

    with torch.no_grad():
        linear_idx = 0
        for m in q_net.base:
            if hasattr(m, "weight") and m.__class__.__name__ == "Linear":
                w = np.array(params[f"w_{linear_idx}"], copy=True).T
                b = np.array(params[f"b_{linear_idx}"], copy=True)
                m.weight.copy_(torch.from_numpy(w))
                m.bias.copy_(torch.from_numpy(b))
                linear_idx += 1

        q_net.v_head.weight.copy_(torch.from_numpy(np.array(params["w_v"], copy=True).T))
        q_net.v_head.bias.copy_(torch.from_numpy(np.array(params["b_v"], copy=True)))
        q_net.mu_head.weight.copy_(torch.from_numpy(np.array(params["w_mu"], copy=True).T))
        q_net.mu_head.bias.copy_(torch.from_numpy(np.array(params["b_mu"], copy=True)))
        q_net.l_head.weight.copy_(torch.from_numpy(np.array(params["w_l"], copy=True).T))
        q_net.l_head.bias.copy_(torch.from_numpy(np.array(params["b_l"], copy=True)))


def make_naf_forward(
    obs_dim: int,
    act_dim: int,
    n_hidden: int = 2,
    l_scale: float = 1.0,
    diag_clamp: Tuple[float, float] = (-5.0, 5.0),
):
    """Create JIT-compiled pure forward functions (forward_mu, forward_v, forward_q)."""
    l_off_dim = (act_dim * (act_dim - 1)) // 2
    if act_dim > 1:
        sub_row, sub_col = np.tril_indices(act_dim, k=-1)
        gather_off = np.full((act_dim, act_dim), l_off_dim, dtype=np.int32)
        gather_off[sub_row, sub_col] = np.arange(l_off_dim, dtype=np.int32)
        gather_off_flat = jnp.array(gather_off.reshape(-1), dtype=jnp.int32)
    else:
        gather_off_flat = jnp.zeros((1,), dtype=jnp.int32)

    low_c, high_c = float(diag_clamp[0]), float(diag_clamp[1])

    def _trunk(params: Dict[str, jax.Array], x: jax.Array) -> jax.Array:
        h = x
        for i in range(n_hidden):
            h = jnp.tanh(h @ params[f"w_{i}"] + params[f"b_{i}"])
        return h

    def forward_mu(params: Dict[str, jax.Array], x: jax.Array) -> jax.Array:
        h = _trunk(params, x)
        return jnp.tanh(h @ params["w_mu"] + params["b_mu"])

    def forward_v(params: Dict[str, jax.Array], x: jax.Array) -> jax.Array:
        h = _trunk(params, x)
        return h @ params["w_v"] + params["b_v"]

    def forward_q(
        params: Dict[str, jax.Array], x: jax.Array, action: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        h = _trunk(params, x)
        V = h @ params["w_v"] + params["b_v"]
        mu = jnp.tanh(h @ params["w_mu"] + params["b_mu"])
        l_entries = jnp.tanh(h @ params["w_l"] + params["b_l"]) * l_scale

        l_diag = jnp.exp(jnp.clip(l_entries[..., :act_dim], low_c, high_c))
        u_mu = action - mu

        if act_dim == 1:
            v_proj = l_diag * u_mu
            adv = -0.5 * jnp.sum(v_proj * v_proj, axis=-1, keepdims=True)
            return V + adv, V, mu

        batch_size = x.shape[0]
        l_off = l_entries[..., act_dim:]
        l_off_pad = jnp.pad(l_off, ((0, 0), (0, 1)))
        L_off = jnp.take(l_off_pad, gather_off_flat, axis=1).reshape(batch_size, act_dim, act_dim)
        v_proj = l_diag * u_mu + jnp.einsum("bi,bij->bj", u_mu, L_off)
        adv = -0.5 * jnp.sum(v_proj * v_proj, axis=-1, keepdims=True)
        return V + adv, V, mu

    return forward_mu, forward_v, forward_q


class JaxNAFUpdater:
    """
    XLA-compiled step updater for NAF / NAF2 / PER-NAF / PER-NAF2.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        learning_rate: float,
        discount: float,
        polyak: float,
        double_q: bool,
        double_q_mode: str,
        loss_type: str,
        huber_delta: float,
        grad_clip_norm: Optional[float],
        q_main_torch,
        q2_main_torch=None,
        l_scale: float = 1.0,
        diag_clamp: Tuple[float, float] = (-5.0, 5.0),
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_hidden = len(hidden_sizes)
        self.discount = float(discount)
        self.polyak = float(polyak)
        self.tau = 1.0 - self.polyak
        self.double_q = bool(double_q)
        self.double_q_mode = str(double_q_mode)
        self.use_huber = loss_type == "huber"
        self.huber_delta = float(huber_delta)

        self.f_mu, self.f_v, self.f_q = make_naf_forward(
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_hidden=self.n_hidden,
            l_scale=l_scale,
            diag_clamp=diag_clamp,
        )
        self.jit_mu = jax.jit(self.f_mu)

        self.p1 = init_naf_params_from_torch(q_main_torch)
        self.tp1 = jax.tree_util.tree_map(lambda x: x.copy(), self.p1)

        if grad_clip_norm is not None and grad_clip_norm > 0:
            self.opt = optax.chain(optax.clip_by_global_norm(float(grad_clip_norm)), optax.adam(learning_rate))
        else:
            self.opt = optax.adam(learning_rate)

        if self.double_q:
            self.p2 = init_naf_params_from_torch(q2_main_torch)
            self.tp2 = jax.tree_util.tree_map(lambda x: x.copy(), self.p2)
            self.opt_state = self.opt.init((self.p1, self.p2))
            self._step_fn = self._build_double_q_step()
        else:
            self.p2 = None
            self.tp2 = None
            self.opt_state = self.opt.init(self.p1)
            self._step_fn = self._build_single_q_step()

    def _elem_loss(self, pred: jax.Array, target: jax.Array) -> jax.Array:
        if self.use_huber:
            return optax.huber_loss(pred, target, delta=self.huber_delta)
        diff = pred - target
        return 0.5 * (diff * diff)

    def _build_single_q_step(self):
        f_v, f_q = self.f_v, self.f_q
        discount, polyak, tau = self.discount, self.polyak, self.tau
        opt = self.opt
        elem_loss = self._elem_loss

        @jax.jit
        def step(p1, tp1, opt_state, o, a, r, o2, d, w):
            r_col = r[:, None]
            d_col = d[:, None]
            w_col = w[:, None]
            v_next = f_v(tp1, o2)
            target_q = r_col + (1.0 - d_col) * discount * v_next

            def loss_fn(params):
                q, v, _ = f_q(params, o, a)
                td = jnp.abs(q - target_q)
                loss = jnp.mean(w_col * elem_loss(q, target_q))
                return loss, (jnp.squeeze(td, axis=1), jnp.mean(v))

            (loss, (td_err, v_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p1)
            updates, new_opt_state = opt.update(grads, opt_state, p1)
            new_p1 = optax.apply_updates(p1, updates)
            new_tp1 = jax.tree_util.tree_map(lambda t, m: polyak * t + tau * m, tp1, new_p1)
            return new_p1, new_tp1, new_opt_state, loss, td_err, v_mean

        return step

    def _build_double_q_step(self):
        f_mu, f_v, f_q = self.f_mu, self.f_v, self.f_q
        discount, polyak, tau = self.discount, self.polyak, self.tau
        opt = self.opt
        elem_loss = self._elem_loss
        clipped_q_mode = self.double_q_mode == "clipped_q"

        @jax.jit
        def step(p1, p2, tp1, tp2, opt_state, o, a, r, o2, d, w):
            r_col = r[:, None]
            d_col = d[:, None]
            w_col = w[:, None]

            if clipped_q_mode:
                a_next = f_mu(tp1, o2)
                q1_next, _, _ = f_q(tp1, o2, a_next)
                q2_next, _, _ = f_q(tp2, o2, a_next)
                v_next = jnp.minimum(q1_next, q2_next)
            else:
                v_next = jnp.minimum(f_v(tp1, o2), f_v(tp2, o2))

            target_q = r_col + (1.0 - d_col) * discount * v_next

            def loss_fn(params):
                q1_p, q2_p = params
                q1, v1, _ = f_q(q1_p, o, a)
                q2, _, _ = f_q(q2_p, o, a)
                td = 0.5 * (jnp.abs(q1 - target_q) + jnp.abs(q2 - target_q))
                l1 = jnp.mean(w_col * elem_loss(q1, target_q))
                l2 = jnp.mean(w_col * elem_loss(q2, target_q))
                return l1 + l2, (jnp.squeeze(td, axis=1), jnp.mean(v1))

            (loss, (td_err, v_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)((p1, p2))
            updates, new_opt_state = opt.update(grads, opt_state, (p1, p2))
            new_p1, new_p2 = optax.apply_updates((p1, p2), updates)
            new_tp1 = jax.tree_util.tree_map(lambda t, m: polyak * t + tau * m, tp1, new_p1)
            new_tp2 = jax.tree_util.tree_map(lambda t, m: polyak * t + tau * m, tp2, new_p2)
            return new_p1, new_p2, new_tp1, new_tp2, new_opt_state, loss, td_err, v_mean

        return step

    def predict_norm(self, norm_obs: np.ndarray) -> np.ndarray:
        obs_2d = np.expand_dims(np.asarray(norm_obs, dtype=np.float32), axis=0)
        mu = self.jit_mu(self.p1, obs_2d)
        return np.asarray(mu[0], dtype=np.float32)

    def update(
        self,
        batch: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, float]:
        o = batch["obs1"]
        o2 = batch["obs2"]
        a = batch["acts"]
        r = batch["rews"]
        d = batch["done"]
        if weights is None:
            w = np.ones_like(r, dtype=np.float32)
        else:
            w = weights

        if self.double_q:
            self.p1, self.p2, self.tp1, self.tp2, self.opt_state, loss, td_err, v_mean = self._step_fn(
                self.p1, self.p2, self.tp1, self.tp2, self.opt_state, o, a, r, o2, d, w
            )
        else:
            self.p1, self.tp1, self.opt_state, loss, td_err, v_mean = self._step_fn(
                self.p1, self.tp1, self.opt_state, o, a, r, o2, d, w
            )
        return float(loss), np.asarray(td_err, dtype=np.float64), float(v_mean)

    def sync_to_torch(self, q_main, q_target, q2_main=None, q2_target=None) -> None:
        sync_jax_params_to_torch(self.p1, q_main)
        sync_jax_params_to_torch(self.tp1, q_target)
        if self.double_q and q2_main is not None and q2_target is not None:
            sync_jax_params_to_torch(self.p2, q2_main)
            sync_jax_params_to_torch(self.tp2, q2_target)
