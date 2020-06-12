import os
import pickle

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pernaf.pernaf.naf import NAF
from pernaf.pernaf.utils.statistic import Statistic
from simple_environment import simpleEnv
# from pendulum import PendulumEnv as simpleEnv
# set random seed
random_seed = 888
# set random seed
tf.set_random_seed(random_seed)
np.random.seed(random_seed)


dof = 5
env = simpleEnv(dof=dof)
env = simpleEnv()
# env = gym.make("Pendulum-v0").env
# env.__name__ = 'Pendulum'
env.seed(random_seed)

for _ in range(10):
    env.reset()

label = 'New NAF_debug on: '+'DOF: '+str(dof) + ' '+ env.__name__

directory = "checkpoints/test_implementation/"

#TODO: Test the loading

def plot_results(env, label):
    # plotting
    print('now plotting')
    rewards = env.rewards
    initial_states = env.initial_conditions

    iterations = []
    finals = []
    starts = []

    # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            finals.append(rewards[i][len(rewards[i]) - 1])
            starts.append(-np.sqrt(np.mean(np.power(initial_states[i], 2))))
            # starts.append(-np.sqrt(np.mean(np.power(initial_states[i], 2))))
            iterations.append(len(rewards[i]))

    plot_suffix = f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    ax=axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)
    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    ax.plot(finals, 'r--')

    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')

    # ax1 = plt.twinx(ax)
    # color = 'lime'
    # ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.plot(starts, color=color)
    plt.savefig(label+'.pdf')
    # fig.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(-np.array(starts), -np.array(finals), c="g", alpha=0.5, marker=r'$\clubsuit$',
                label="Luck")
    plt.ylim(0, 3)
    plt.title(label)
    plt.show()

def plot_convergence(agent, label):
    losses, vs = agent.losses, agent.vs
    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel('episodes')

    color = 'tab:blue'
    ax.semilogy(losses, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'

    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(vs, color=color)
    plt.savefig(label + 'convergence' + '.pdf')
    plt.show()


if __name__ == '__main__':

    discount = 0.999
    batch_size = 10
    learning_rate = 1e-3
    max_steps = 200
    update_repeat = 5
    max_episodes = 100
    tau = 1 - 0.999
    is_train = True
    is_continued = False

    nafnet_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.tanh
                         , weight_init=tf.random_uniform_initializer(-0.05, 0.05))

    noise_info = dict(noise_function = lambda nr: max(0.00001, 1-nr/100))

    prio_info = dict(alpha=.95, beta_start=.9, beta_decay = lambda nr: max(0.00001, 1-nr/25))
    # prio_info = dict()
    # filename = 'Scan_data.obj'
    # filehandler = open(filename, 'rb')
    # scan_data = pickle.load(filehandler)


    with tf.Session() as sess:
        # statistics and running the agent
        stat = Statistic(sess=sess, env_name=env.__name__, model_dir=directory,
                         max_update_per_step=update_repeat, is_continued=is_continued, save_frequency=500)
        # init the agent
        agent = NAF(sess=sess, env=env, stat=stat, discount= discount, batch_size=batch_size,
                    learning_rate=learning_rate, max_steps=max_steps, update_repeat=update_repeat,
                    max_episodes=max_episodes, tau=tau, pretune = None, prio_info=prio_info,
                    noise_info=noise_info, **nafnet_kwargs)
        # run the agent
        agent.run(is_train)

    # plot the results
    plot_convergence(agent=agent, label=label)
    plot_results(env, label)

