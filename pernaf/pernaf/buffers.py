
import numpy as np
from .utils.prioritised_experience_replay import PrioritizedReplayBuffer


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for NAF_debug agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        if self.size < batch_size:
            idxs = np.arange(self.size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class ReplayBufferPER(PrioritizedReplayBuffer):
    """
    A simple FIFO experience replay buffer for NAF_debug agents.
    """

    def __init__(self, obs_dim, act_dim, size, prio_info):
        self.alpha = prio_info.get('alpha')
        self.beta = prio_info.get('beta')
        super(ReplayBufferPER, self).__init__(size, self.alpha)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        super(ReplayBufferPER, self).add(obs, act, rew, next_obs, done, 1)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        if self.size < batch_size:
            batch_size = self.size
            obs1, acts, rews, obs2, done, gammas, weights, idxs = super(ReplayBufferPER, self).sample_normal(batch_size)
        else:
            obs1, acts, rews, obs2, done, gammas, weights, idxs = super(ReplayBufferPER, self).sample(batch_size,
                                                                                                      self.beta)
        return dict(obs1=obs1,
                    obs2=obs2,
                    acts=acts,
                    rews=rews,
                    done=done), [weights, idxs]

