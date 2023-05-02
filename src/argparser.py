import argparse

def parse_args():
    """
    Parses command line arguments
    """
    parser = argparse.ArgumentParser(description='Parse terminal input information')

    n_episodes = 2000 # number of episodes
    N = 2 # number of players
    BS = 64 # batch size
    ponderated_avg = 100 # ponderated average size
    auction = 'first_price' # auction type
    z = 1 # number of executions
    save_plot = 1 # save plot
    alert = 0 # alert

    parser.add_argument('-e', '--episodes', type=int, help='Total number of training episodes')
    parser.add_argument('-n', '--players', type=int, help='Total number of players')
    parser.add_argument('-b', '--batch', type=int, help='Batch size')
    parser.add_argument('-p', '--ponderated', type=int, help='Ponderated average size')
    parser.add_argument('-a', '--auction', type=str, help='Auction type')
    parser.add_argument('-z', '--executions', type=int, help='Number of executions')
    parser.add_argument('-s', '--save', type=int, help='Save plot')
    parser.add_argument('-t', '--alert', type=int, help='Alert')
    args = parser.parse_args()

    # if arguments are passed, overwrite default values
    if args.episodes:
        n_episodes = args.episodes
    if args.players:
        N = args.players
    if args.batch:
        BS = args.batch
    if args.ponderated:
        ponderated_avg = args.ponderated
    if args.auction:
        auction = args.auction
    if args.executions:
        z = args.executions
    if args.save:
        save_plot = args.save
    if args.alert:
        alert = args.alert

    print('Number of episodes: ', n_episodes)
    print('Number of players: ', N)
    print('Batch size: ', BS)
    print('Ponderated average size: ', ponderated_avg)
    print('Auction type: ', auction)
    print('Number of executions: ', z)
    print('Save plot: ', save_plot)
    print('Alert: ', alert)
    print('\n')

    return n_episodes, N, BS, ponderated_avg, auction, z, save_plot, alert