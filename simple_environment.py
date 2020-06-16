import logging.config
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt
import gym
import numpy as np
# 3rd party modules
import math
from enum import Enum


class simpleEnv(gym.Env):
    """
    Define a simple environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, **kwargs):
        self.__version__ = "0.0.1"
        logging.info("simple_ENV - Version {}".format(self.__version__))
        self.__name__ = "simple_ENV - Version {}".format(self.__version__)
        # General variables defining the environment
        self.is_finalized = False
        self.MAX_TIME = 100
        self.curr_step = -1

        self.curr_episode = -1
        self.TOTAL_COUNTER = -1
        self.action_episode_memory = []
        self.rewards = []
        self.initial_conditions = []

        self.counter = 0
        # self.seed(123)
        if 'dof' in kwargs:
            self.dimension = kwargs.get('dof')
        else:
            self.dimension = 10

        self.MAX_POS = 2
        self.action_space = gym.spaces.Box(low=-self.MAX_POS, high=self.MAX_POS, shape=(self.dimension,),
                                           dtype=np.float32)
        print('Action space dim is: ', self.action_space)

        # Create observation space
        self.MAX_POS = 1
        self.observation_space = gym.spaces.Box(low=-self.MAX_POS, high=self.MAX_POS, shape=(self.dimension,),
                                                dtype=np.float32)

        self.reference_trajectory = np.ones(self.dimension)
        self.response_matrix = np.eye(self.dimension) * ((np.random.uniform(-0.5,-0.25, self.dimension)))

        self.delta_abs = np.zeros(self.action_space.shape[0])
        print('State space dim is: ', self.observation_space)
        print(self.response_matrix)

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        self.curr_step += 1
        self.counter += 1
        state, reward = self._take_action(action)
        self.action_episode_memory[self.curr_episode].append(action)
        self.rewards[self.curr_episode].append(reward)
        if reward < - 100 or reward > -.1 or self.curr_step > self.MAX_TIME:
            self.is_finalized = True

        # if self.is_finalized:
        #     print('Finished at:\n', state, reward, self.curr_step)

        return state, reward, self.is_finalized, {}

    def _take_action(self, delta_action):
        # print('d ', self.delta_abs)
        abs_action = delta_action+ self.delta_abs
        self.TOTAL_COUNTER += 1
        state = np.dot(self.response_matrix, abs_action)
        reward = np.sqrt(np.mean(np.square(state - self.reference_trajectory)))
        self.delta_abs = abs_action.copy()
        print('step', self.curr_step)
        print('a ', self.delta_abs)
        print('d ', delta_action)
        print('s ', state)
        print('r', reward)
        return state, -reward

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.curr_episode += 1
        self.curr_step = 0

        self.action_episode_memory.append([])
        self.rewards.append([])

        self.is_finalized = False
        self.delta_abs = np.random.uniform(-.5,.5,self.dimension)
        init_state, init_reward = self._take_action(np.zeros(self.action_space.shape[0]))
        print('inits ', init_reward)
        self.initial_conditions.append(init_state)
        return init_state


if __name__ == '__main__':
    environment_instance = simpleEnv(dof=2)
    # print(environment_instance.reset())
    # action = -np.ones(environment_instance.action_space.shape)
    # starts = []
    # ends = []
    # for _ in range(20):
    #     starts.append(environment_instance.reset())
    #     environment_instance.step(action=action)
    #     ends.append(environment_instance.step(action=action)[0])
    # starts = np.array(starts)
    # ends = np.array(ends)
    # plt.scatter(starts[:,0], starts[:,1])
    # plt.scatter(ends[:, 0], ends[:, 1], c='r')
    # plt.show()
    # environment_instance.reset()
    # for _ in range(100):
    #     print(environment_instance.step(np.random.uniform(low=-1, high=1, size=environment_instance.action_space.shape[0]))[1])

    rews = []
    actions = []
    states = []

    zero_action =[]
    def objective(action):
        actions.append(action.copy())
        if len(zero_action)>0:
            delta_action =  action-zero_action[-1]
        else:
            delta_action = action
        # print('obj:', delta_action, action)
        state, r, f, _ = environment_instance.step(action=delta_action)
        zero_action.append(action.copy())
        # print(zero_action)
        rews.append(r)
        return -r


    # print(environment_instance.reset())


    def constr(action):
        if any(action > environment_instance.action_space.high[0]):
            return -1
        elif any(action < environment_instance.action_space.low[0]):
            return -1
        else:
            return 1

    results = []
    for _ in range(1):
        start_vector = environment_instance.reset()
        # print('init: ', environment_instance.reset())
        start_vector = np.zeros(environment_instance.action_space.shape[0])
        rhobeg = environment_instance.action_space.high[0]
        # print('rhobeg: ', rhobeg)
        res = opt.fmin_cobyla(objective, start_vector, [constr], rhobeg=rhobeg, rhoend=.1, disp=0)
        # res = opt.fmin_powell(objective, start_vector)
        # print(res)
        # print(np.sqrt(np.sum(np.square(res))))
        results.append(res)
        print(res)
        print(objective(res))




