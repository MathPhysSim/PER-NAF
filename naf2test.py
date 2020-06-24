import os
import pickle
import random

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from naf2 import NAF
from pernaf.pernaf.utils.statistic import Statistic
from simple_environment import simpleEnv
# from pendulum import PendulumEnv as simpleEnv
# set random seed
random_seed = 111
# set random seed

np.random.seed(random_seed)
random.seed(random_seed)


dof = 2
env = simpleEnv(dof=dof)
# env = simpleEnv()
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
    final_rews = []
    starts = []
    sum_rews=[]
    mean_rews = []
    # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            final_rews.append(rewards[i][len(rewards[i]) - 1])
            starts.append(-np.sqrt(np.mean(np.square(initial_states[i]))))
            iterations.append(len(rewards[i]))
            sum_rews.append(np.sum(rewards[i]))
            mean_rews.append(np.mean(rewards[i]))
    plot_suffix = ""#f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    ax=axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)
    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    ax.plot(final_rews, 'r--')

    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.set_ylabel('starts', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(starts, color=color)
    plt.savefig(label+'.pdf')
    # fig.tight_layout()
    plt.show()


    fig, axs = plt.subplots(1, 1)
    axs.plot(sum_rews)
    ax1 = plt.twinx(axs)
    ax1.plot(mean_rews,c='lime')
    plt.show()

def plot_convergence(agent, label):
    losses, vs = agent.losses, agent.vs
    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel('# steps')

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
    batch_size = 5
    learning_rate = 1e-3
    max_steps = 50
    update_repeat = 5
    max_episodes = 20
    polyak = 0.999
    is_train = True
    is_continued = False

    nafnet_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.tanh
                         , weight_init=tf.random_uniform_initializer(-0.05, 0.05, seed=random_seed))

    noise_info = dict(noise_function = lambda nr: max(0, (1-nr/10)))

    prio_info = dict(alpha=.5, beta=.9)
    prio_info = dict()

    # filename = 'Scan_data.obj'
    # filehandler = open(filename, 'rb')
    # scan_data = pickle.load(filehandler)

    # init the agent
    agent = NAF(env=env, discount= discount, batch_size=batch_size,
                learning_rate=learning_rate, max_steps=max_steps, update_repeat=update_repeat,
                max_episodes=max_episodes, polyak=polyak, pretune = None, prio_info=prio_info,
                noise_info=noise_info, **nafnet_kwargs)
    # run the agent
    agent.run(is_train)

    # plot the results
    plot_convergence(agent=agent, label=label)
    plot_results(env, label)

