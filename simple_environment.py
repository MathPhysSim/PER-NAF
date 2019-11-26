import logging.config
import random

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
        self.MAX_TIME = 50
        self.curr_step = -1

        self.curr_episode = -1
        self.TOTAL_COUNTER = -1
        self.action_episode_memory = []
        self.rewards = []
        self.initial_conditions = []

        self.counter = 0
        self.seed(123)
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
        self.response_matrix = np.eye(self.dimension) * (np.clip(np.random.randn(self.dimension),-0.5,0.5)+1)

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
        if reward < - 10 or reward > -.5 or self.curr_step > self.MAX_TIME:
            self.is_finalized = True

        # if self.is_finalized:
        #     print('Finished at:\n', state, reward)
        return state, reward, self.is_finalized, {}

    def _take_action(self, action):
        self.TOTAL_COUNTER += 1
        state = np.dot(self.response_matrix, action)
        reward = np.sqrt(np.mean(np.square(state - self.reference_trajectory)))
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
        init_state, init_reward = self._take_action(np.random.randn(self.dimension))
        self.initial_conditions.append(init_state)
        return init_state


if __name__ == '__main__':
    env = simpleEnv()
    print(env.reset())
    action = np.ones(env.action_space.shape)
    print(env.step(action=action))
