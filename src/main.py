import timeit
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound
from datetime import timedelta

# Local modules
from agent import Agent
from env import FirstPriceAuctionEnv, SecondPriceAuctionEnv
from utils import plotLearning, manualTesting
from argparser import parse_args
from train import trainLoop

n_episodes, N, BS, ponderated_avg, auction, z, save_plot, alert = parse_args() # get parameters
for k in range(1, z+1):
    if auction == 'first_price':
        env = FirstPriceAuctionEnv(N)
    else:
        env = SecondPriceAuctionEnv(N)
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[1], tau=0.001, env=env,
                batch_size=BS,  layer1_size=400, layer2_size=400, n_actions=1)

    score_history = trainLoop(agent, env, n_episodes, ponderated_avg, N, BS, k)
    if alert == 1:
        playsound('stuff/beep.mp3')
    # plotLearning(score_history, 'results/' + auction +  '_auction_ ' + str(n_episodes) + '.png', window=100)    
    if save_plot == 1:
        manualTesting(agent, N, k, n_episodes, auc_type=auction)