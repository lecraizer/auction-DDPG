# General modules
import timeit
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound
from datetime import timedelta

# Local modules
from agent import Agent
from multiagent_env import MAFirstPriceAuctionEnv, MASecondPriceAuctionEnv, MACommonPriceAuctionEnv
from utils import plotLearning, manualTesting
from argparser import parse_args
from train import trainLoop, MAtrainLoop, MAtrainLoopCommonValue

k = 0
n_episodes, N, BS, ponderated_avg, auction, z, save_plot, alert = parse_args() # get parameters

if auction == 'first_price':
    multiagent_env = MAFirstPriceAuctionEnv(N)
elif auction == 'second_price':
    multiagent_env = MASecondPriceAuctionEnv(N)
elif auction == 'common_value':
    multiagent_env = MACommonPriceAuctionEnv(N)

agents = [Agent(alpha=0.000025, beta=0.00025, input_dims=[1], tau=0.001, env=multiagent_env,
            batch_size=BS,  layer1_size=400, layer2_size=400, n_actions=1) for i in range(N)]

if auction == 'common_value':
    score_history = MAtrainLoopCommonValue(agents, multiagent_env, n_episodes, auction)
else:
    score_history = MAtrainLoop(agents, multiagent_env, n_episodes, auction)
if alert == 1:
    playsound('stuff/beep.mp3')