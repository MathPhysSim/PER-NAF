import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow_core.python.keras.losses import MSE

from pernaf.pernaf.utils.prioritised_experience_replay import PrioritizedReplayBuffer

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

def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(y_true - y_pred)


def fc(x, hidden_size, name=None):
    layer = keras.layers.Dense(hidden_size, activation=tf.tanh,
                               kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.05, 0.05),
                               kernel_regularizer=None,
                               bias_initializer=tf.compat.v1.constant_initializer(0.0), name=name)
    return layer(x)


obs_dim = 2
act_dim = 2
# action = tf.Variable(np.ones(act_dim), dtype=float)
hidden_sizes = (100, 100)


class QModel():

    def __init__(self, obs_dim=2, act_dim=2, hidden_sizes=(100, 100), **kwargs):
        self.hidden_sizes = hidden_sizes
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # create a shared network for the variables
        inputs = keras.Input(shape=(obs_dim + act_dim,))
        h = inputs[:, 0:obs_dim]
        for hidden_dim in hidden_sizes:
            h = fc(h, hidden_dim)
        V = fc(h, 1, name='V')
        l = fc(h, (act_dim * (act_dim + 1) / 2))
        mu = fc(h, act_dim, name='mu')

        action = inputs[:, obs_dim:]
        pivot = 0
        rows = []
        for idx in range(act_dim):
            count = act_dim - idx
            diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
            non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
            row = tf.pad(tensor=tf.concat((diag_elem, non_diag_elems), 1), paddings=((0, 0), (idx, 0)))
            rows.append(row)
            pivot += count
        L = tf.transpose(a=tf.stack(rows, axis=1), perm=(0, 2, 1))
        P = tf.matmul(L, tf.transpose(a=L, perm=(0, 2, 1)))
        tmp = tf.expand_dims(action - mu, -1)
        A = -tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]), tf.matmul(P, tmp)) / 2
        A = tf.reshape(A, [-1, 1])
        Q = A + V

        self.q_model = keras.Model(inputs=inputs, outputs=Q)
        self.q_model.compile(keras.optimizers.Adam(learning_rate=0.0002), loss=MSE)

        # Action output
        self.model_get_action = keras.Model(inputs=self.q_model.layers[0].input,
                                            outputs=self.q_model.get_layer(name='mu').output)

        # Value output
        self.model_value_estimate = keras.Model(inputs=self.q_model.layers[0].input,
                                                outputs=self.q_model.get_layer(name='V').output)

    def get_action(self, state):
        actions = np.zeros((state.shape[0], act_dim))
        input = np.concatenate((state, actions), axis=1)
        print(input)
        return self.model_get_action.predict(input)

    def get_value_estimate(self, state):
        actions = np.zeros((state.shape[0], act_dim))
        input = np.concatenate((state, actions), axis=1)
        return self.model_value_estimate.predict(input)

    def set_weights(self, weights, polyak = 0.999):
        old_weights = self.get_weights()
        weights = [polyak * old_weights[i] + (1-polyak) * weights[i] for i in range(len(weights))]
        self.q_model.set_weights(weights=weights)

    def get_weights(self):
        return self.q_model.get_weights()

    def train_model(self, batch_s, batch_a, batch_y):
        batch_x = np.concatenate((batch_s, batch_a), axis=1)
        hist = q_main_model.q_model.fit(batch_x, batch_y)
        return hist.history['loss']






class NAF(object):
    def __init__(self, sess,
                 env, stat,
                 discount, batch_size, learning_rate,
                 max_steps, update_repeat, max_episodes, tau, pretune = None, prio_info=dict(), noise_info=dict(), **nafnet_kwargs):
        '''
        :param sess: current tensorflow session
        :param env: open gym environment to be solved
        :param stat: statistic class to handle tensorflow and statitics
        :param discount: discount factor
        :param batch_size: batch size for the training
        :param learning_rate: learning rate
        :param max_steps: maximal steps per episode
        :param update_repeat: iteration per step of training
        :param max_episodes: maximum number of episodes
        :param tau: polyac averaging
        :param pretune: list of tuples of state action reward next state done
        :param prio_info: parameters to handle the prioritizing of the buffer
        :param nafnet_kwargs: keywords to handle the network
        :param noise_info: dict with noise_function
        '''
        self.pretune = pretune
        self.prio_info = prio_info
        self.per_flag = bool(self.prio_info)
        print('PER is:', self.per_flag)

        self.sess = sess
        self.env = env

        if 'noise_function' in noise_info:
            self.noise_function = noise_info.get('noise_function')
        else:
            self.noise_function = lambda nr: 1/(nr+1)

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

        self.q_main_model = QModel(obs_dim=self.obs_dim, act_dim=self.action_size)
        self.q_target_model = QModel(obs_dim=self.obs_dim, act_dim=self.action_size)

    def predict(self, state, is_train):
        u = self.q_main_model.get_action(state=state)
        if is_train:
            noise_scale = self.noise_function(self.idx_episode)
            return u + noise_scale * np.random.randn(self.action_size)
        else:
            return u

    def run(self, is_train=True):
        # pretune------------------------------------------------------------------------------
        if not(self.pretune is None):
            scan_data = self.pretune
            print('Length of scan data is: ', len(scan_data))

            if scan_data:
                for i, data in enumerate(scan_data):
                    o, a, r, o2, d, _ = data
                    self.replay_buffer.store(o, a, r, o2, d)
                    # print("Number: ", i)
                    # print(o, a, r, o2, d)

            batch_size_temp = self.batch_size
            self.batch_size = 10
            for _ in range(10*len(scan_data)):
                q, v, a, l = self.update_q()
                if self.stat:
                    self.stat.on_step(a, r, d, q, v, a, l)

            self.batch_size = batch_size_temp
        # -------------------------------------------------------------------------------------
        for self.idx_episode in range(self.max_episodes):
            o = self.env.reset()
            for t in range(0, self.max_steps):
                # 1. predict
                a = self.predict(o, is_train)
                # 2. interact
                o2, r, d, _ = self.env.step(a)
                if is_train:
                    self.replay_buffer.store(o, a, r, o2, d)
                o = o2
                d = False if t == self.max_steps - 1 else d
                # 3. perceive
                if is_train:
                    pass
                    self.update_q()
                if d:
                    break

    def update_q(self):
        for iteration in range(self.update_repeat):
            if self.per_flag:
                batch, priority_info = self.replay_buffer.sample_batch(self.batch_size)
            else:
                batch = self.replay_buffer.sample_batch(self.batch_size)

            o = batch['obs1']
            o2 = batch['obs2']
            a = batch['acts']
            r = batch['rews']
            d = batch['done']

            v = self.q_target_model.get_value_estimate(o2, a)
            target_y = self.discount * np.squeeze(v)*(1-d) + r
            loss = self.q_main_model.train_model(o, a, target_y)

            if self.per_flag:
                priorities = np.ones(priority_info[0].shape[-1]) * (loss * 1 + 1e-7)
                self.replay_buffer.update_priorities(idxes=priority_info[1], priorities=priorities)

        self.sess.run(self.target_update)
        self.vs.append(np.mean(v))

        # return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list)



if __name__ == '__main__':

    # test_state = np.random.random((1, 2))
    #
    # q_main_model = QModel(2, 2)
    # q_target_model = QModel(2, 2)
    #
    # print('main', q_main_model.get_action(test_state))
    # print('main', q_main_model.get_value_estimate(test_state))
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    # q_target_model.set_weights(q_main_model.get_weights())
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    # batch_x = np.random.random((5, 4))
    # batch_y = np.random.random((5, 4))
    # hist = q_main_model.q_model.fit(batch_x, batch_y)
    # print(hist.history['loss'])
    #
    # print('main', q_main_model.get_action(test_state))
    # print('main', q_main_model.get_value_estimate(test_state))
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    #
    # q_target_model.set_weights(q_main_model.get_weights())
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    # weights = (q_target_model.get_weights())
    # keras.utils.plot_model(model, 'my_first_model.png')
    # keras.utils.plot_model(model_get_action, 'model_get_action.png')


