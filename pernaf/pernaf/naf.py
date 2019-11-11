from logging import getLogger

from .utils import core

logger = getLogger(__name__)


import numpy as np
import tensorflow as tf
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


class NAF(object):
    def __init__(self, sess,
                 env, stat,
                 discount, batch_size, learning_rate,
                 max_steps, update_repeat, max_episodes, tau, pretune = None, prio_info=dict(), **nafnet_kwargs):

        self.pretune = pretune
        self.prio_info = prio_info
        self.per_flag = bool(self.prio_info)
        print('PER is:', self.per_flag)
        self.sess = sess
        self.env = env

        self.x_ph, self.a_ph, self.mu, self.V, self.Q, self.P, self.A, self.vars_pred \
            = core.mlp_normalized_advantage_function(env.observation_space.shape, act_dim=env.action_space.shape,
                                                     **nafnet_kwargs,
                                                     scope='main')
        self.x_ph_targ, self.a_ph_targ, self.mu_targ, self.V_targ, self.Q_targ, self.P_targ, self.A_targ, \
        self.vars_targ \
            = core.mlp_normalized_advantage_function(env.observation_space.shape, act_dim=env.action_space.shape,
                                                     **nafnet_kwargs,
                                                     scope='target')
        self.stat = stat
        self.discount = discount
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.action_size = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]

        self.max_steps = max_steps
        self.update_repeat = update_repeat
        self.max_episodes = max_episodes

        if not (self.per_flag):
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(1e6))
        else:
            self.replay_buffer = ReplayBufferPER(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(1e6),
                                                 prio_info=prio_info)

        with tf.name_scope('optimizer'):
            self.target_y = tf.placeholder(tf.float32, [None], name='target_y')
            self.per_weights = tf.placeholder(tf.float32, [None], name='per_weights')
            # self.loss = tf.reduce_mean(tf.squared_difference(self.target_y, tf.squeeze(self.Q)),
            #                            name='loss')
            self.loss = tf.losses.mean_squared_error(self.target_y, tf.squeeze(self.Q), weights=self.per_weights)
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.gvs = self.optim.compute_gradients(self.loss)
            # self.modified_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
            # self.train_op = self.optim.apply_gradients(self.modified_gvs)

        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                     for v_main, v_targ in zip(self.vars_pred, self.vars_targ)])

        # Polyak averaging for target variables (previous soft update)
        polyak = 1 - tau
        self.target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1 - polyak) * v_main)
                                       for v_main, v_targ in zip(self.vars_pred, self.vars_targ)])
        self.losses = []
        self.vs = []

    def run(self, is_train=True):
        print('Training:', is_train)
        # tf.initialize_all_variables().run()
        self.stat.set_variables(self.vars_pred)
        self.stat.load_model()  # including init
        # if is_train:
        #     self.sess.run(self.target_init)

        # pretune------------------------------------------------------------------------------
        if not(self.pretune is None):
            scan_data = self.pretune
            print('Length of scan data is: ', len(scan_data))

            if scan_data:
                for i, data in enumerate(scan_data):
                    o, a, r, o2, d, _ = data
                    self.replay_buffer.store(o, a, r, o2, d)
                    print("Number: ", i)
                    print(o, a, r, o2, d)

            batch_size_temp = self.batch_size
            self.batch_size = 10
            for _ in range(10*len(scan_data)):
                q, v, a, l = self.perceive()
                if self.stat:
                    self.stat.on_step(a, r, d, q, v, a, l)

            self.batch_size = batch_size_temp
        # -------------------------------------------------------------------------

        for self.idx_episode in range(self.max_episodes):
            o = self.env.reset()
            for t in range(0, self.max_steps):
                # 1. predict
                a = self.predict(o, is_train)
                # 2. step
                o2, r, d, _ = self.env.step(a)
                if is_train:
                    self.replay_buffer.store(o, a, r, o2, d)
                o = o2
                d = False if t == self.max_steps - 1 else d
                # 3. perceive
                if is_train:
                    pass
                    q, v, a, l = self.perceive()
                    if self.stat:
                        self.stat.on_step(a, r, d, q, v, a, l)
                if d:
                    break

    def predict(self, state, is_train):
        u = self.sess.run(self.mu, feed_dict={self.x_ph: [state]})[0]
        if is_train:
            noise_scale = 1 / (self.idx_episode + 1)
            return u + noise_scale * np.random.randn(self.action_size)
        else:
            return u

    def perceive(self):
        q_list = []
        v_list = []
        a_list = []
        l_list = []

        for iteration in range(self.update_repeat):
            if self.per_flag:
                batch, priority_info = self.replay_buffer.sample_batch(self.batch_size)
            else:
                batch = self.replay_buffer.sample_batch(self.batch_size)

            o = batch['obs1']
            o2 = batch['obs2']
            a = batch['acts']
            r = batch['rews']
            w = priority_info[0]

            v = self.sess.run(self.V_targ, feed_dict={self.x_ph_targ: o2, self.a_ph_targ: a})
            target_y = self.discount * np.squeeze(v) + r

            _, l, q, v, a = self.sess.run([
                self.optim, self.loss,
                self.Q, self.V, self.A,
            ], {
                self.target_y: target_y,
                self.x_ph: o,
                self.a_ph: a,
                self.per_weights: w
            })

            q_list.extend(q)
            v_list.extend(v)
            a_list.extend(a)
            l_list.append(l)

            # self.target_network.soft_update_from(self.pred_network)

            if self.per_flag:
                priorities = np.ones(priority_info[0].shape[-1]) * (abs(l) * 1 + 1e-7)
                self.replay_buffer.update_priorities(idxes=priority_info[1], priorities=priorities)

            logger.debug("q: %s, v: %s, a: %s, l: %s" \
                         % (np.mean(q), np.mean(v), np.mean(a), np.mean(l)))
        self.sess.run(self.target_update)
        self.losses.append(np.mean(l))
        self.vs.append(np.mean(v))

        return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list)
