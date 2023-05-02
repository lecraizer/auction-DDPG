import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import timeit


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