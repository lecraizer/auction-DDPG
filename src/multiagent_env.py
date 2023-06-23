import random
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box


class MAFirstPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players
        self.values = [random.random() for _ in range(self.N)]
        

    def reward_n_players(self, values, bids):
        rewards = [0]*self.N
        idx = np.argmax(bids)
        rewards[idx] = values[idx] - bids[idx]
        return rewards

        
    def step(self, states, actions):
        rewards = self.reward_n_players(states, actions)
        
        # End episode
        done = True
        info = {} # set placeholder for info

        # Return step information
        return rewards, done

    def reset(self):
        # Reset state - input new random private value
        self.values = [random.random() for _ in range(self.N)]
        return self.values
    

class MASecondPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players
        self.values = [random.random() for _ in range(self.N)]
        

    def reward_n_players(self, values, bids):
        rewards = [0]*self.N
        end_list = np.argsort(bids)[-2:]
        second_max_idx = end_list[0]
        max_idx = end_list[1]
        rewards[max_idx] = values[max_idx] - bids[second_max_idx]
        return rewards

        
    def step(self, states, actions):
        rewards = self.reward_n_players(states, actions)
        
        # End episode
        done = True
        info = {} # set placeholder for info

        # Return step information
        return rewards, done

    def reset(self):
        # Reset state - input new random private value
        self.values = [random.random() for _ in range(self.N)]
        return self.values
    

class MACommonPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players
        self.values = [random.random() for _ in range(self.N)]
        

    def reward_n_players(self, common_value, bids):
        rewards = [0]*self.N
        idx = np.argmax(bids)
        rewards[idx] = common_value - bids[idx]
        return rewards

        
    def step(self, state, actions):
        rewards = self.reward_n_players(state, actions)
        
        # End episode
        done = True
        info = {} # set placeholder for info

        # Return step information
        return rewards, done

    def reset(self):
        # Reset state - input new random common value
        self.common_value = random.random()
        # list of signals for each agent in which signal is a normal distribution with mean = common_value and std = 1
        self.signals = [np.random.normal(self.common_value, 1) for _ in range(self.N)] 

        return self.common_value, self.signals