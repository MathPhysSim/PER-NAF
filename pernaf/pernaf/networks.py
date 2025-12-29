
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(100, 100), activation=nn.Tanh, **kwargs):
        super(QNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Shared feature extractor (Common trunk)
        # In original code, h = fc(h, hidden_dim) was sequential.
        # But 'inputs' was (obs + act). Wait, NAF usually separates V(s) and A(s,a).
        # Original: inputs = (obs_dim + act_dim). 
        # h = inputs[:, 0:obs_dim] <- Slices ONLY OBS!
        # So the network is purely state-dependent until the end.
        
        layers = []
        input_dim = obs_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(activation())
            input_dim = h_dim
            
        self.base = nn.Sequential(*layers)
        
        # Value Stream: State -> V
        self.v_head = nn.Linear(input_dim, 1)
        
        # Action Mean Stream: State -> Mu
        self.mu_head = nn.Linear(input_dim, act_dim)
        
        # Lower-Triangular Matrix Stream: State -> L Entries
        # Number of entries in L: act_dim * (act_dim + 1) / 2
        self.l_entries_dim = int(act_dim * (act_dim + 1) / 2)
        self.l_head = nn.Linear(input_dim, self.l_entries_dim)
        
        # Initialization (mimic original uniform -0.05, 0.05 if desired, or let PyTorch default)
        # Original: random_uniform_initializer(-0.05, 0.05)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, action=None):
        # x is observation
        features = self.base(x)
        
        V = self.v_head(features)
        mu = self.mu_head(features)
        mu = torch.tanh(mu) # Standard practice usually involves tanh for action bounds? Original code didn't explicitly show tanh on mu output in the extracted snippet, but NAF mu is usually the action.
        # Wait, original snippet: "outputs=self.q_model.get_layer(name='mu').output"
        # And "l = self.fc(h, ...)" where self.fc had activation=tf.tanh.
        # BUT the head itself? "mu = self.fc(h, act_dim, name='mu')". yes, self.fc HAS tanh.
        # So Mu is tanh activated.
        
        l_entries = self.l_head(features) # Also has tanh from self.fc
        
        # Reconstruct L matrix
        # This is tricky in batch mode without explicit loops or fancy indexing.
        batch_size = x.size(0)
        L = torch.zeros((batch_size, self.act_dim, self.act_dim), device=x.device)
        
        # Fill diagonal (exponentiated) and lower triangle
        # We need to map l_entries to L
        # L_entries layout: [diag_0, (diag_1, sub_1_0), (diag_2, sub_2_0, sub_2_1), ...]
        
        current_idx = 0
        for i in range(self.act_dim):
            # Diagonal
            L[:, i, i] = torch.exp(l_entries[:, current_idx])
            current_idx += 1
            # Lower diagonal
            for j in range(i):
                L[:, i, j] = l_entries[:, current_idx]
                current_idx += 1
                
        # P = L * L^T
        P = torch.bmm(L, L.transpose(1, 2))
        
        Q = None
        if action is not None:
            # A = -0.5 * (u - mu)^T * P * (u - mu)
            u_mu = (action - mu).unsqueeze(2) # (Batch, Act, 1)
            u_mu_T = u_mu.transpose(1, 2)     # (Batch, 1, Act)
            
            # (Batch, 1, Act) * (Batch, Act, Act) -> (Batch, 1, Act)
            adv_temp = torch.bmm(u_mu_T, P)
            # (Batch, 1, Act) * (Batch, Act, 1) -> (Batch, 1, 1)
            advantage = -0.5 * torch.bmm(adv_temp, u_mu).squeeze(2) # (Batch, 1)
            
            Q = V + advantage
            
        return Q, V, mu, P

