import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
from tensorflow.contrib.layers import fully_connected


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def fc(layer, num_outputs, scope='fc'):
    batch_norm_args = {}
    with tf.variable_scope(scope):
        return fully_connected(
            layer,
            num_outputs=int(num_outputs),
            activation_fn=tf.tanh,
            weights_initializer=tf.random_uniform_initializer(-0.05, 0.05),
            weights_regularizer=None,
            biases_initializer=tf.constant_initializer(0.0),
            # scope=scope,
            ** batch_norm_args
        )

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Normalize Advantage Function
"""


def mlp_normalized_advantage_function(x, act_dim, hidden_sizes=(100,100), activation=tf.tanh,
                                      output_activation=tf.tanh, action_space=None, weight_init=None,
                                      act_multiplier=1, scope=None):
    with tf.variable_scope(scope):
        # act_dim = a.shape.as_list()[-1]
        act_dim = act_dim[0]
        # act_limit = action_space.high[0] * act_multiplier

        x_ph = tf.placeholder(tf.float32, (None,) + tuple(x), name='observations')
        a_ph = tf.placeholder(tf.float32, (None, act_dim), name='actions')

        # create a shared network for the variables
        with tf.name_scope('hidden'):
            h = x_ph
            for idx, hidden_dim in enumerate(hidden_sizes):
                h = fc(h, hidden_dim, scope='hid%d' % idx)

        with tf.name_scope('value'):
            V = fc(h, 1, scope='V')

        with tf.name_scope('advantage'):
            l = fc(h, (act_dim * (act_dim + 1) / 2), scope='l')
            mu = fc(h, act_dim, scope='mu')

            pivot = 0
            rows = []
            for idx in range(act_dim):
                count = act_dim - idx
                diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
                non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
                row = tf.pad(tf.concat((diag_elem, non_diag_elems), 1), ((0, 0), (idx, 0)))
                rows.append(row)
                pivot += count
            L = tf.transpose(tf.stack(rows, axis=1), (0, 2, 1))
            P = tf.matmul(L, tf.transpose(L, (0, 2, 1)))
            tmp = tf.expand_dims(a_ph - mu, -1)
            A = -tf.matmul(tf.transpose(tmp, [0, 2, 1]), tf.matmul(P, tmp)) / 2
            A = tf.reshape(A, [-1, 1])

        with tf.name_scope('Q'):
            Q = A + V

            # print(mu.name, V.name, Q.name, P.name, A.name, h.name)
        vars = get_variables(scope)
        return x_ph, a_ph, mu, V, Q, P, A, vars


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    tf.set_random_seed(123)
    np.random.seed(123)

    MAX_POS = 1
    action_space = gym.spaces.Box(low=-MAX_POS, high=MAX_POS, shape=(1,),
                                  dtype=np.float32)
    observation_space = gym.spaces.Box(low=-MAX_POS, high=MAX_POS, shape=(1,),
                                       dtype=np.float32)

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    nafnet_kwargs = dict(hidden_sizes=[1, 1], activation=tf.tanh,
                         weight_init=tf.random_uniform_initializer(-0.05, 0.05))
    # Share information about action space with pernaf architecture
    nafnet_kwargs['action_space'] = action_space

    # Inputs to computation graph
    # x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)
    x_ph, a_ph = placeholders(obs_dim, act_dim)
    x2_ph, r_ph, d_ph = placeholders(obs_dim, None, None)
    # Main outputs from computation graph
    with tf.variable_scope('main'):

        print('generate main network')
        mu_pred, V_pred, Q_pred, P_pred, A_pred = \
            mlp_normalized_advantage_function(x_ph, a_ph, **nafnet_kwargs)
    # Target networks
    with tf.variable_scope('target'):
        print('generate target network')
        _, V_targ, _, _, _ = mlp_normalized_advantage_function(x2_ph, a_ph, **nafnet_kwargs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    test_vector = [[i] for i in range(-100, 100)]
    a = sess.run([mu_pred, V_pred, Q_pred, P_pred, A_pred], feed_dict={x_ph: test_vector,
                                     a_ph: test_vector})
    plt.plot(np.squeeze(a[0]))
    plt.plot(np.squeeze(a[1]))
    plt.title('Mu_pred - me')
    plt.show()

