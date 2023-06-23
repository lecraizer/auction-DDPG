import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import timeit
import random
from utils import plotLearning, manualTesting

def trainLoop(agent, env, n_episodes, ponderated_avg, N, BS, k):
    # agent.load_models()
    np.random.seed(0)
    start_time = timeit.default_timer()

    score_history = []
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        print('\nEpisode', i)
        print('Value: ', obs)
        act = agent.choose_action(obs)
        print('Bid:   ', act[0])
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        score_history.append(score)
        print('Score   %.2f' % score,
            'trailing ' + str(ponderated_avg) + ' games avg %.3f' % np.mean(score_history[-ponderated_avg:]))
        
        # if i % 25 == 0:
        #    agent.save_models()

    # Total training time
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])
    return score_history


def MAtrainLoop(agents, env, n_episodes, auction_type='first_price'):
    # agent.load_models()
    np.random.seed(0)
    start_time = timeit.default_timer()
    N = len(agents)
    for ep in range(n_episodes):
        observations = env.reset()
        done = False
        
        # fixed agent 0
        for idx in range(N):
            original_actions = [agents[i].choose_action(observations[i])[0] for i in range(N)]

            for new_action in np.linspace(0.001, 0.999, 100):
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards, done = env.step(observations, actions)
                agents[idx].remember(observations[idx], actions[idx], rewards[idx], observations[idx], int(done))
                agents[idx].learn()

        done = True
        if ep % 1 == 0:
            print('\nEpisode', ep)
            print('Values:  ', observations)
            print('Bids:    ', [agents[i].choose_action(observations[0])[0] for i in range(len(agents))])
            print('Rewards: ', rewards)
            for i in range(len(agents)):
                manualTesting(agents[i], N, 'ag'+str(i+1), n_episodes, auc_type=auction_type)

    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])


def MAtrainLoopCommonValue(agents, env, n_episodes, auction_type='first_price'):
    np.random.seed(0)
    start_time = timeit.default_timer()
    N = len(agents)
    for ep in range(n_episodes):
        common_value, observations = env.reset()
        done = False
        
        # fixed agent 0
        for idx in range(N):
            original_actions = [agents[i].choose_action(observations[i])[0] for i in range(N)]

            for new_action in np.linspace(0.001, 0.999, 100):
                actions = original_actions[:idx] + [new_action] + original_actions[idx+1:]
                rewards, done = env.step(common_value, actions)
                agents[idx].remember(observations[idx], actions[idx], rewards[idx], observations[idx], int(done))
                agents[idx].learn()

        done = True
        if ep % 1 == 0:
            print('\nEpisode', ep)
            print('Value:  ', common_value)
            print('Signals: ', observations)
            print('Bids:    ', [agents[i].choose_action(observations[0])[0] for i in range(len(agents))])
            print('Rewards: ', rewards)
            for i in range(len(agents)):
                manualTesting(agents[i], N, 'ag'+str(i+1), n_episodes, auc_type=auction_type)

    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])









########## ------------ OLD ------------ ##########


def old_MAtrainLoop(agents, env, n_episodes, ponderated_avg, N, BS, k):
    # agent.load_models()
    np.random.seed(0)
    start_time = timeit.default_timer()

    score_history = []
    for i in range(n_episodes):
        observations = env.reset()
        done = False
        score = 0
        print('\nEpisode', i)
        print('Values: ', observations)
        actions = []
        for i in range(len(agents)):
            action = agents[i].choose_action(observations[i])
            actions.append(action)
        print('Bid 1: ', actions[0])
        print('Bid 2: ', actions[1])

        new_states, rewards, done, info = env.step(actions)
        for i in range(len(agents)):
            agents[i].remember(observations[i], actions[i], rewards[i], new_states[i], int(done))
            agents[i].learn()
        
        score += rewards[0]
        observations = new_states
        score_history.append(score)
        print('Score   %.2f' % score,
            'trailing ' + str(ponderated_avg) + ' games avg %.3f' % np.mean(score_history[-ponderated_avg:]))
        
        # if i % 25 == 0:
        #    agent.save_models()

    # Total training time
    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])
    return score_history


def MAtrainLoop_2N(agents, env, n_episodes, ponderated_avg, N, BS, k):
    # agent.load_models()
    np.random.seed(0)
    start_time = timeit.default_timer()

    score_history = []
    for ep in range(n_episodes):
        observations = env.reset()
        done = False
        
        # fixed agent
        action0 = agents[0].choose_action(observations[0])[0]

        for action1 in np.linspace(0.001, 0.999, 100):
            # action1 = agents[1].choose_action(obs)[0]
            actions = [action0, action1]

            rewards, done = env.step(observations, actions)
            agents[1].remember(observations[1], actions[1], rewards[1], observations[1], int(done))
            agents[1].learn()

        # fixed another agent
        action1 = agents[1].choose_action(observations[1])[0]

        for action0 in np.linspace(0.001, 0.999, 100):
            actions = [action0, action1]

            rewards, done = env.step(observations, actions)
            agents[0].remember(observations[0], actions[0], rewards[0], observations[0], int(done))
            agents[0].learn()
        

        done = True
        if ep % 50 == 0:
            print('\nEpisode', ep)
            print('Values:  ', observations)
            print('Bids:    ', [agents[i].choose_action(observations[0])[0] for i in range(len(agents))])
            print('Rewards: ', rewards)
            manualTesting(agents[0], N, 'ag1', n_episodes, auc_type='first_price')
            manualTesting(agents[1], N, 'ag2', n_episodes, auc_type='first_price')

    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])
    # return score_history


def MAtrainLoop_old(agents, env, n_episodes, ponderated_avg, N, BS, k):
    # agent.load_models()
    np.random.seed(0)
    start_time = timeit.default_timer()

    for ep in range(n_episodes):
        observations = env.reset()
        done = False
        
        # fixed agent 0
        idx = 0

        action1 = agents[1].choose_action(observations[1])[0]
        action2 = agents[2].choose_action(observations[2])[0]

        for action0 in np.linspace(0.001, 0.999, 100):
            actions = [action0, action1, action2]
            rewards, done = env.step(observations, actions)
            agents[0].remember(observations[0], actions[0], rewards[0], observations[0], int(done))
            agents[0].learn()

        # fixed another agent
        action0 = agents[0].choose_action(observations[0])[0]
        action2 = agents[2].choose_action(observations[2])[0]

        for action1 in np.linspace(0.001, 0.999, 100):
            actions = [action0, action1, action2]

            rewards, done = env.step(observations, actions)
            agents[1].remember(observations[1], actions[1], rewards[1], observations[1], int(done))
            agents[1].learn()
        
        # fixed another agent
        action0 = agents[0].choose_action(observations[0])[0]
        action1 = agents[1].choose_action(observations[1])[0]

        for action2 in np.linspace(0.001, 0.999, 100):
            actions = [action0, action1, action2]

            rewards, done = env.step(observations, actions)
            agents[2].remember(observations[2], actions[2], rewards[2], observations[2], int(done))
            agents[2].learn()


        done = True
        if ep % 1 == 0:
            print('\nEpisode', ep)
            print('Values:  ', observations)
            print('Bids:    ', [agents[i].choose_action(observations[0])[0] for i in range(len(agents))])
            print('Rewards: ', rewards)
            for i in range(len(agents)):
                manualTesting(agents[i], N, 'ag'+str(i+1), n_episodes, auc_type='first_price')

    total_time = timeit.default_timer() - start_time
    print('\n\nTotal training time: ', str(timedelta(seconds=total_time)).split('.')[0])
    # return score_history

