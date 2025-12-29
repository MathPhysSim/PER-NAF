
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .networks import QNetwork
from .buffers import ReplayBuffer, ReplayBufferPER


class NAF(object):
    """
    Normalized Advantage Function (NAF) Agent using PyTorch.
    Supports Double Q-Learning (NAF2).
    """
    def __init__(self, env, learning_rate=1e-3, 
                 buffer_size=1000000, 
                 batch_size=10, 
                 discount=0.999, 
                 polyak=0.999, 
                 max_steps=100, 
                 update_repeat=3,
                 prio_info=dict(),
                 noise_info=dict(), 
                 directory=None, 
                 device="cpu",
                 double_q=False,
                 **nafnet_kwargs):
        
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount = discount
        self.polyak = polyak
        self.max_steps = max_steps
        self.update_repeat = update_repeat
        self.prio_info = prio_info
        self.noise_info = noise_info
        self.directory = directory
        self.nafnet_kwargs = nafnet_kwargs
        self.double_q = double_q
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"NAF Agent (DoubleQ={self.double_q}) running on: {self.device}")

        # Logging / Stats
        self.losses = []
        self.vs = []
        self.episode_rewards = []
        self.counter = 0

        # PER Setup
        self.per_flag = bool(self.prio_info)
        print('PER is:', self.per_flag)

        if 'noise_function' in noise_info:
            self.noise_function = noise_info.get('noise_function')
        else:
            self.noise_function = lambda nr: 1 / (nr + 1)

        self.action_size = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]

        # Init Buffers
        if not (self.per_flag):
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(buffer_size))
        else:
            self.replay_buffer = ReplayBufferPER(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(buffer_size),
                                                 prio_info=prio_info)
        
        if 'decay_function' in prio_info:
            self.decay_function = prio_info.get('decay_function')
        else:
            if 'beta' in prio_info:
                self.decay_function = lambda nr: prio_info.get('beta')
            else:
                self.decay_function = lambda nr: 1.

        # Init Networks
        self.q_main = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
        self.q_target = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q_main.state_dict())
        
        # Double Q Init
        if self.double_q:
            self.q2_main = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
            self.q2_target = QNetwork(self.obs_dim, self.action_size, **nafnet_kwargs).to(self.device)
            self.q2_target.load_state_dict(self.q2_main.state_dict())
            
            # Joint Optimizer
            self.optimizer = optim.Adam(list(self.q_main.parameters()) + list(self.q2_main.parameters()), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.q_main.parameters(), lr=learning_rate)
        
        if self.directory and not os.path.exists(self.directory):
             os.makedirs(self.directory)

    def predict(self, observation, state=None, deterministic=False):
        self.q_main.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            _, _, mu, _ = self.q_main(obs_tensor)
            action = mu.cpu().numpy()[0]
            
        self.q_main.train()
        return action, state

    def learn(self, total_timesteps, callback=None, log_interval=100):
        steps = 0
        episodes = 0
        
        o, info = self.env.reset() # Gymnasium returns (obs, info)
        
        pbar = tqdm(total=total_timesteps)
        
        while steps < total_timesteps:
            episodes += 1
            noise_scale = self.noise_function(episodes)
            episode_reward = 0
            
            done = False
            while not done and steps < total_timesteps:
                # 1. Predict
                action_mean, _ = self.predict(o, deterministic=True) # predict handles eval mode
                action = action_mean + noise_scale * np.random.randn(self.action_size)
                action = np.clip(action, -1, 1) # Optional Clip if tanh is used?
                
                # 2. Step
                o2, r, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += r
                
                # Store
                self.replay_buffer.store(o, action, r, o2, done)
                
                o = o2
                steps += 1
                pbar.update(1)
                
                # 3. Train
                if self.replay_buffer.size > self.batch_size:
                    for _ in range(self.update_repeat):
                        self.update_q(episodes)
                     
            self.episode_rewards.append(episode_reward)
            if done:
                o, _ = self.env.reset()

        pbar.close()
        return self

    def update_q(self, episode_idx=0):
        self.counter += 1
        decay = self.decay_function(episode_idx)
        
        # Sample
        if self.per_flag:
            batch, priority_info = self.replay_buffer.sample_batch(self.batch_size)
        else:
            batch = self.replay_buffer.sample_batch(self.batch_size)

        o = torch.FloatTensor(batch['obs1']).to(self.device)
        o2 = torch.FloatTensor(batch['obs2']).to(self.device)
        a = torch.FloatTensor(batch['acts']).to(self.device)
        r = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device) # (B, 1)
        d = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device) # (B, 1)

        # Compute Target Q
        with torch.no_grad():
            if self.double_q:
                # Double Q: Min of Target Values
                _, V_next_1, _, _ = self.q_target(o2)
                _, V_next_2, _, _ = self.q2_target(o2)
                V_next = torch.min(V_next_1, V_next_2)
            else:
                _, V_next, _, _ = self.q_target(o2)
                
            target_q = r + (1 - d) * self.discount * V_next

        if self.double_q:
            # Q1
            Q1, _, _, _ = self.q_main(o, a)
            # Q2
            Q2, _, _, _ = self.q2_main(o, a)
            
            # Loss
            if self.per_flag:
                weights = torch.FloatTensor(priority_info[0]).unsqueeze(1).to(self.device)
                loss1 = (weights * F.mse_loss(Q1, target_q, reduction='none')).mean()
                loss2 = (weights * F.mse_loss(Q2, target_q, reduction='none')).mean()
                loss = loss1 + loss2
            else:
                loss1 = F.mse_loss(Q1, target_q)
                loss2 = F.mse_loss(Q2, target_q)
                loss = loss1 + loss2
        else:
            # Standard NAF
            Q, _, _, _ = self.q_main(o, a)
            if self.per_flag:
                weights = torch.FloatTensor(priority_info[0]).unsqueeze(1).to(self.device)
                loss = (weights * F.mse_loss(Q, target_q, reduction='none')).mean()
            else:
                loss = F.mse_loss(Q, target_q)
            
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update Priorities if PER
        if self.per_flag:
             if self.double_q:
                 # Use average Q or just Q1? Q1 is 'main'.
                 error = torch.abs(Q1 - target_q).detach().cpu().numpy()
             else:
                 error = torch.abs(Q - target_q).detach().cpu().numpy()
                 
             new_priorities = (error + 1e-7).flatten()
             self.replay_buffer.update_priorities(idxes=priority_info[1], priorities=new_priorities)

        # Polyak Averaging
        with torch.no_grad():
            for target_param, param in zip(self.q_target.parameters(), self.q_main.parameters()):
                target_param.data.copy_(target_param.data * self.polyak + param.data * (1.0 - self.polyak))
            
            if self.double_q:
                for target_param, param in zip(self.q2_target.parameters(), self.q2_main.parameters()):
                    target_param.data.copy_(target_param.data * self.polyak + param.data * (1.0 - self.polyak))

        # Log
        self.losses.append(loss.item())
        if self.double_q:
             self.vs.append(Q1.mean().item()) # Tracking Q1 value
        else:
             self.vs.append(Q.mean().item())

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(path))
        torch.save(self.q_main.state_dict(), path + ".pth")
        if self.double_q:
             torch.save(self.q2_main.state_dict(), path + "_q2.pth")
        print(f"Model saved to {path}.pth")
    
    def load(self, path):
        self.q_main.load_state_dict(torch.load(path + ".pth", map_location=self.device))
        self.q_target.load_state_dict(self.q_main.state_dict())
        if self.double_q:
             if os.path.exists(path + "_q2.pth"):
                  self.q2_main.load_state_dict(torch.load(path + "_q2.pth", map_location=self.device))
                  self.q2_target.load_state_dict(self.q2_main.state_dict())
        print(f"Model loaded from {path}.pth")

