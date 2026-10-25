"""
Normalized Advantage Function (NAF) neural network architecture in PyTorch.

Implements a zero-loop, non-mutating Cholesky advantage decomposition:
    Q(s, a) = V(s) + A(s, a)
    A(s, a) = -0.5 * (a - mu(s))^T P(s) (a - mu(s))
            = -0.5 * || L(s)^T (a - mu(s)) ||_2^2
where P(s) = L(s) L(s)^T and L(s) = diag(exp(l_diag(s))) + L_strict_lower(s).
"""

from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_activation(activation: Union[str, type, nn.Module, callable]) -> nn.Module:
    """Resolve activation specification (class, instance, string, or legacy callable) to nn.Module."""
    if isinstance(activation, nn.Module):
        return type(activation)()
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()
    if isinstance(activation, str):
        name = activation.lower()
        mapping = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if name in mapping:
            return mapping[name]()
    if callable(activation):
        name = getattr(activation, "__name__", "").lower()
        if "tanh" in name:
            return nn.Tanh()
        if "relu" in name:
            return nn.ReLU()
        if "elu" in name:
            return nn.ELU()
    return nn.Tanh()


class QNetwork(nn.Module):
    """
    Vectorized Normalized Advantage Function (NAF) Q-Network.

    Provides specialized fast paths:
    - ``forward_mu(x)``: computes only the greedy action ``mu(s)`` in [-1, 1].
    - ``forward_v(x)``: computes only the state-value ``V(s)``.
    - ``forward_q(x, action)``: computes ``(Q(s, a), V(s), mu(s))`` in O(B * d_a^2)
      using ``index_select`` and ``||L^T (a - mu)||_2^2`` without in-place slice assignment
      or forming P = L @ L^T.
    - ``forward(x, action=None, return_p=False)``: full forward pass for backward compatibility.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (100, 100),
        activation: Union[str, type, nn.Module] = nn.Tanh,
        init_bound: float = 0.05,
        layer_norm: bool = False,
        l_scale: float = 1.0,
        diag_clamp: Tuple[float, float] = (-5.0, 5.0),
        **kwargs,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        self.init_bound = float(init_bound)
        self.layer_norm = bool(layer_norm)
        self.l_scale = float(l_scale)
        self.diag_clamp = diag_clamp

        layers = []
        input_dim = self.obs_dim
        for h_dim in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, int(h_dim)))
            if self.layer_norm:
                layers.append(nn.LayerNorm(int(h_dim)))
            layers.append(_resolve_activation(activation))
            input_dim = int(h_dim)

        self.base = nn.Sequential(*layers)

        # Value stream: State -> V(s)
        self.v_head = nn.Linear(input_dim, 1)

        # Action mean stream: State -> mu(s) in [-1, 1]
        self.mu_head = nn.Linear(input_dim, self.act_dim)

        # Cholesky lower-triangular factor stream:
        # First `act_dim` entries are diagonal; remaining `act_dim*(act_dim-1)//2` are strict lower triangle
        self.l_off_dim = (self.act_dim * (self.act_dim - 1)) // 2
        self.l_entries_dim = self.act_dim + self.l_off_dim
        self.l_head = nn.Linear(input_dim, self.l_entries_dim)

        # Precompute flat index_select map for strict lower-triangular entries (pointing to zero pad at l_off_dim)
        if self.act_dim > 1:
            sub_row, sub_col = torch.tril_indices(row=self.act_dim, col=self.act_dim, offset=-1)
            gather_off = torch.full((self.act_dim, self.act_dim), self.l_off_dim, dtype=torch.long)
            gather_off[sub_row, sub_col] = torch.arange(self.l_off_dim, dtype=torch.long)
            self.register_buffer("gather_off_flat", gather_off.view(-1), persistent=False)
        else:
            self.register_buffer("gather_off_flat", torch.zeros(1, dtype=torch.long), persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -self.init_bound, self.init_bound)
            nn.init.constant_(m.bias, 0.0)

    def forward_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path: compute only the greedy action mu(s) in [-1, 1]."""
        features = self.base(x)
        return torch.tanh(self.mu_head(features))

    def forward_v(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path: compute only the state value V(s)."""
        features = self.base(x)
        return self.v_head(features)

    def _build_l_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Construct batch of full lower-triangular matrices L(s) with positive diagonal."""
        batch_size = features.size(0)
        l_entries = torch.tanh(self.l_head(features)) * self.l_scale
        l_diag_raw = l_entries[:, : self.act_dim]
        if self.diag_clamp is not None:
            l_diag_raw = torch.clamp(l_diag_raw, self.diag_clamp[0], self.diag_clamp[1])
        l_diag = torch.exp(l_diag_raw)

        if self.act_dim == 1:
            return l_diag.unsqueeze(2)

        l_off_pad = F.pad(l_entries[:, self.act_dim :], (0, 1))
        L = l_off_pad.index_select(1, self.gather_off_flat).view(batch_size, self.act_dim, self.act_dim)
        return L + torch.diag_embed(l_diag)

    def forward_q(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast path for training: compute Q(s, a), V(s), mu(s) in O(B * d_a^2)
        using ||L^T (a - mu)||_2^2 without in-place mutation or constructing P = L @ L^T.
        """
        features = self.base(x)
        V = self.v_head(features)
        mu = torch.tanh(self.mu_head(features))
        l_entries = torch.tanh(self.l_head(features)) * self.l_scale

        l_diag_raw = l_entries[:, : self.act_dim]
        if self.diag_clamp is not None:
            l_diag_raw = torch.clamp(l_diag_raw, self.diag_clamp[0], self.diag_clamp[1])
        l_diag = torch.exp(l_diag_raw)

        u_mu = action - mu
        if self.act_dim == 1:
            v_proj = l_diag * u_mu
            advantage = -0.5 * (v_proj * v_proj)
            return V + advantage, V, mu

        batch_size = features.size(0)
        l_off_pad = F.pad(l_entries[:, self.act_dim :], (0, 1))
        L_off = l_off_pad.index_select(1, self.gather_off_flat).view(batch_size, self.act_dim, self.act_dim)
        v_proj = l_diag * u_mu + torch.bmm(u_mu.unsqueeze(1), L_off).squeeze(1)
        advantage = -0.5 * (v_proj * v_proj).sum(dim=1, keepdim=True)
        return V + advantage, V, mu

    def forward(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        return_p: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Full forward pass returning (Q, V, mu, P).
        """
        if action is not None and not return_p:
            Q, V, mu = self.forward_q(x, action)
            return Q, V, mu, None

        features = self.base(x)
        V = self.v_head(features)
        mu = torch.tanh(self.mu_head(features))
        L = self._build_l_matrix(features)
        P = torch.bmm(L, L.transpose(1, 2)) if (return_p or action is None) else None

        Q = None
        if action is not None:
            u_mu = (action - mu).unsqueeze(1)
            v_proj = torch.bmm(u_mu, L).squeeze(1)
            advantage = -0.5 * (v_proj * v_proj).sum(dim=1, keepdim=True)
            Q = V + advantage

        return Q, V, mu, P
