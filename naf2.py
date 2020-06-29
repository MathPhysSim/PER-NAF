import os
import random
import time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_docs as tfdocs
# from tensorflow_core.python.keras.losses import MSE
from tensorflow.python.keras.losses import MSE
from tqdm import tqdm

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


# obs_dim = 2
# act_dim = 2
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
            h = self.fc(h, hidden_dim)
        V = self.fc(h, 1, name='V')

        l = self.fc(h, (act_dim * (act_dim + 1) / 2))
        mu = self.fc(h, act_dim, name='mu')

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
        A = -tf.multiply(tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]), tf.matmul(P, tmp)), tf.constant(0.5))
        A = tf.reshape(A, [-1, 1])
        Q = tf.add(A, V)

        self.learning_rate = 1e-3
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs.get('learning_rate')
            # print('learning rate', self.learning_rate )
        if 'directory' in kwargs:
            self.directory = kwargs.get('directory')
        else:
            self.directory = None

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.q_model = keras.Model(inputs=inputs, outputs=Q)
        # self.q_model.compile(keras.optimizers.Adam(learning_rate=self.learning_rate), loss=MSE)
        self.q_model.compile(optimizer="adam", loss="mse", metrics=["mae"], )
        # Action output
        self.model_get_action = keras.Model(inputs=self.q_model.layers[0].input,
                                            outputs=self.q_model.get_layer(name='mu').output)

        # Value output
        self.model_value_estimate = keras.Model(inputs=self.q_model.layers[0].input,
                                                outputs=self.q_model.get_layer(name='V').output)

        # self.q_model.summary()

    def fc(self, x, hidden_size, name=None):
        layer = keras.layers.Dense(hidden_size, activation=tf.tanh,
                                   kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.05, 0.05),
                                   kernel_regularizer=None,
                                   bias_initializer=tf.compat.v1.constant_initializer(0.0), name=name)
        return layer(x)

    def get_action(self, state):
        state = np.array([state])
        actions = np.zeros((state.shape[0], self.act_dim))
        # print('actions: ', actions)
        input = np.concatenate((state, actions), axis=1)
        # print('input: ', input)
        return self.model_get_action.predict(input)

    def get_value_estimate(self, state):
        actions = np.zeros((state.shape[0], self.act_dim))
        input = np.concatenate((state, actions), axis=1)
        return self.model_value_estimate.predict(input)

    def set_polyak_weights(self, weights, polyak=0.999):
        weights_old = self.get_weights()
        weights_new = [polyak * weights_old[i] + (1 - polyak) * weights[i] for i in range(len(weights))]
        self.q_model.set_weights(weights=weights_new)

    def get_weights(self):
        return self.q_model.get_weights()

    def save_model(self, directory):
        try:
            self.q_model.save(filepath=directory, overwrite=True)
        except:
            print('Saving failed')

    def train_model(self, batch_s, batch_a, batch_y, **kwargs):
        # batch_x = np.concatenate((batch_s, batch_a), axis=1)
        x_batch_train = tf.keras.layers.concatenate([batch_s, batch_a],
                                                    axis=1, dtype='float32')
        y_batch_train = tf.convert_to_tensor(batch_y, dtype='float32')

        epochs = 1000
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        batch_size = 10
        hist = self.q_model.fit(x_batch_train, y_batch_train,
                                validation_split=0.2,
                                verbose=4,
                                # batch_size=batch_size,
                                epochs=epochs,
                                callbacks=[callback], **kwargs)
        return hist.history['loss']

    # def train_model(self, batch_s, batch_a, y_batch_train, **kwargs):
    #     # x_batch_train = np.concatenate((batch_s, batch_a), axis=1)
    #     x_batch_train = tf.keras.layers.concatenate([batch_s, batch_a],
    #                                                 axis=1, dtype='float64')
    #     y_batch_train = tf.convert_to_tensor(y_batch_train, dtype='float64')
    #
    #     return self.train_step(x_batch_train, y_batch_train, **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x_batch_train, y_batch_train, **kwargs):

        # Iterate over the batches of the dataset.
        if 'sample_weight' in kwargs:
            sample_weight = kwargs.get('sample_weight')
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            targets = self.q_model(x_batch_train, training=True)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            loss_value = tf.multiply(keras.losses.MSE(y_batch_train, targets), sample_weight)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.q_model.trainable_weights)
        # grads()
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_weights))
        return loss_value

    class CustomModel(keras.Model):
        def train_step(self, data):
            # Unpack the data. Its structure depends on your model and
            # on what you pass to `fit()`.
            print(data)
            x, y = data

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}


class NAF(object):
    def __init__(self, env, discount, batch_size, learning_rate,
                 max_steps, update_repeat, max_episodes, polyak=0.999, pretune=None, prio_info=dict(),
                 noise_info=dict(), save_frequency=500, directory=None, is_continued=False, **nafnet_kwargs):
        '''
        :param env: open gym environment to be solved
        :param directory: directory were weigths are saved
        :param stat: statistic class to handle tensorflow and statitics
        :param discount: discount factor
        :param batch_size: batch size for the training
        :param learning_rate: learning rate
        :param max_steps: maximal steps per episode
        :param update_repeat: iteration per step of training
        :param max_episodes: maximum number of episodes
        :param polyak: polyac averaging
        :param pretune: list of tuples of state action reward next state done
        :param prio_info: parameters to handle the prioritizing of the buffer
        :param nafnet_kwargs: keywords to handle the network
        :param noise_info: dict with noise_function
        '''
        self.losses2 = []
        self.vs2 = []
        self.directory = directory
        self.save_frequency = save_frequency
        self.polyak = polyak
        self.losses = []
        self.pretune = pretune
        self.prio_info = prio_info
        self.per_flag = bool(self.prio_info)
        print('PER is:', self.per_flag)

        self.env = env

        if 'noise_function' in noise_info:
            self.noise_function = noise_info.get('noise_function')
        else:
            self.noise_function = lambda nr: 1 / (nr + 1)

        self.discount = discount
        self.batch_size = batch_size

        self.action_size = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]

        self.max_steps = max_steps
        self.update_repeat = update_repeat
        self.max_episodes = max_episodes
        self.idx_episode = None
        self.vs = []

        if not (self.per_flag):
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(1e6))
            self.replay_buffer2 = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(1e6))

        else:
            self.replay_buffer = ReplayBufferPER(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(1e6),
                                                 prio_info=prio_info)
            self.replay_buffer2 = ReplayBufferPER(obs_dim=self.obs_dim, act_dim=self.action_size, size=int(1e6),
                                                  prio_info=prio_info)
        if 'decay_function' in prio_info:
            self.decay_function = prio_info.get('decay_function')
        else:
            if 'beta' in prio_info:
                self.decay_function = lambda nr: prio_info.get('beta')
            else:
                self.decay_function = lambda nr: 1.

        self.q_main_model_1 = QModel(obs_dim=self.obs_dim, act_dim=self.action_size, learning_rate=learning_rate)
        self.q_main_model_2 = QModel(obs_dim=self.obs_dim, act_dim=self.action_size, learning_rate=learning_rate)
        # Set same initial values in all networks
        # self.q_main_model_2.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())
        # Set same initial values in all networks
        self.q_target_model_1 = QModel(obs_dim=self.obs_dim, act_dim=self.action_size)
        self.q_target_model_1.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())
        self.q_target_model_2 = QModel(obs_dim=self.obs_dim, act_dim=self.action_size)
        self.q_target_model_2.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())

        if is_continued:
            try:
                # self.q_main.q_model = tf.keras.models.load_model(filepath=self.directory)
                self.q_target_model_1.q_model = tf.keras.models.load_model(filepath=self.directory)
                print('loaded', 10 * ' -')
            except:
                print('failed', 10 * ' *')
                if not os.path.exists(self.directory):
                    os.makedirs(self.directory)
        else:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            elif not (self.directory):
                for f in os.listdir(self.directory):
                    print('Deleting: ', self.directory + '/' + f)
                    os.remove(self.directory + '/' + f)
                time.sleep(.5)

        self.counter = 0

    def predict(self, state, is_train):

        if is_train:
            action = self.q_main_model_1.get_action(state=state)
            noise_scale = self.noise_function(self.idx_episode)
            return action + noise_scale * np.random.randn(self.action_size)
        else:
            action = self.q_target_model_1.get_action(state=state)
            return action

    def run(self, is_train=True):
        for index in tqdm(range(0, self.max_episodes)):
            self.idx_episode = index
            o = self.env.reset()

            for t in range(0, self.max_steps):
                # print(self.idx_episode, t)
                # 1. predict
                a = self.predict(o, is_train)[0]
                # 2. interact
                o2, r, d, _ = self.env.step(a)
                if is_train:
                    self.replay_buffer.store(o, a, r, o2, d)
                    self.replay_buffer2.store(o, a, r, o2, d)
                o = o2
                d = False if t == self.max_steps - 1 else d
                # 3. perceive
                if is_train and self.replay_buffer.size > 1:
                    try:
                        self.update_q()
                    except:
                        print('wait:', self.replay_buffer.size)
                if d:
                    break

    def update_q(self):
        vs = []
        losses = []
        vs2 = []
        losses2 = []
        self.counter += 1
        decay = self.decay_function(self.idx_episode)

        for model in [1 if random.randint(0, 1) < 0.5 else 2]:
            for iteration in range(self.update_repeat):
                if model == 1:
                    if self.per_flag:
                        batch, priority_info = self.replay_buffer.sample_batch(self.batch_size)
                    else:
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                else:
                    if self.per_flag:
                        batch, priority_info = self.replay_buffer2.sample_batch(self.batch_size)
                    else:
                        batch = self.replay_buffer.sample_batch(self.batch_size)

                o = batch['obs1']
                o2 = batch['obs2']
                a = batch['acts']
                r = batch['rews']
                d = batch['done']

                v_1 = self.q_target_model_1.get_value_estimate(o2)
                v_2 = self.q_target_model_2.get_value_estimate(o2)
                v = np.where(v_1 < v_2, v_1, v_2)
                target_y = self.discount * np.squeeze(v) * (1 - d) + r
                # target_y = self.discount * np.squeeze(v) + r
                # print(target_y)
                if model == 1:
                    if self.per_flag:
                        # TODO: change to tensorflow
                        sample_weights = tf.convert_to_tensor(priority_info[0] * decay, dtype='float32')
                        loss = self.q_main_model_1.train_model(o, a, target_y, sample_weight=sample_weights)[-1]
                    else:
                        loss = self.q_main_model_1.train_model(o, a, target_y)[-1]
                    vs.append(v)
                    losses.append(loss)
                    if self.per_flag:
                        # TODO: change to tensorflow
                        sample_weights = tf.convert_to_tensor(priority_info[0] * decay, dtype='float32')
                        priorities = np.ones(priority_info[0].shape[-1]) * (loss + 1e-7)
                        self.replay_buffer.update_priorities(idxes=priority_info[1], priorities=priorities)
                else:
                    if self.per_flag:
                        loss = self.q_main_model_2.train_model(o, a, target_y, sample_weight=sample_weights)[-1]
                    else:
                        loss = self.q_main_model_2.train_model(o, a, target_y)[-1]
                    vs2.append(v)
                    losses2.append(loss)
                    if self.per_flag:
                        priorities = np.ones(priority_info[0].shape[-1]) * (loss + 1e-7)
                        self.replay_buffer2.update_priorities(idxes=priority_info[1], priorities=priorities)

                # print('loss :', loss)
        if model == 1:
            self.q_target_model_1.set_polyak_weights(self.q_main_model_1.get_weights(), polyak=self.polyak)
        else:
            self.q_target_model_2.set_polyak_weights(self.q_main_model_2.get_weights(), polyak=self.polyak)

        # print('ep:', self.idx_episode, 'loss :', np.mean(losses))
        # if random.uniform(0, 1) < 0.5:
        #     # print('network 1', np.mean(v_1))
        #     self.q_target_model_1.set_polyak_weights(self.q_main_model_1.get_weights(), polyak=self.polyak)
        # else:
        #     self.q_target_model_2.set_polyak_weights(self.q_main_model_1.get_weights(), polyak=self.polyak)
        # print('network 2',  np.mean(v_2))
        if self.counter % self.save_frequency == 0:
            # print('saving: ', self.counter)
            self.q_target_model_1.save_model(directory=self.directory)
        self.vs.append(np.mean(vs))
        self.losses.append(np.mean(losses))
        self.vs2.append(np.mean(vs2))
        self.losses2.append(np.mean(losses2))


if __name__ == '__main__':
    print('start')
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
