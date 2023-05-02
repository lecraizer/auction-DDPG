import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)


def manualTesting(agent, N, k, n_episodes, auc_type='first_price'):
    # reset plot variables
    plt.close('all')

    states = np.linspace(0, 1, 100) # private values
    actions = []
    avg_error = 0
    for state in states:
        action = agent.choose_action(state)[0] # bid
        if auc_type == 'first_price':
            expected_action = state*(N-1)/N
        else:
            expected_action = state
        avg_error += abs(action - expected_action)
        actions.append(action)
    avg_error /= len(states)
    print('Average error: %.3f' % avg_error)

    # plt scatter size small
    plt.scatter(states, actions, color='black', s=0.3)
    if auc_type == 'first_price':
        plt.plot(states, states*(N-1)/N, color='brown', linewidth=0.5)
        plt.title('First Price Auction for ' + str(N) + ' players')
    else:
        plt.plot(states, states, color='brown', linewidth=0.5)
        plt.title('Second Price Auction for ' + str(N) + ' players')
    plt.text(0.02, 0.94, 'Avg error: %.3f' % avg_error, fontsize=10, color='#696969')
    plt.legend(['Expected bid', 'Agent bid'], loc='lower right')   
    plt.xlabel('State (Value)')
    plt.ylabel('Action (Bid)')

    # set x-axis and y-axis range to [0, 1]
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    plt.savefig('results/' + auc_type + '/N=' + str(N) + '/test' + str(int(n_episodes/1000)) + 'k_' + str(k) + '.png')