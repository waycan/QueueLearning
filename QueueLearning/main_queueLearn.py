import numpy as np
import argparse
from System import System
import matplotlib.pyplot as plt
import time, os, sys

class queueLearn_time(object):
    _parameters = {'num_server': 0, 'num_class': 0, 'time_horizon': 0, 'num_tsk_per_usr': 0, 'mu': None,
                   'Lambda': 0, 'class_prob': None, 'reward_mat': None, 'V': 0, 'gamma': 0, 'verbose': False,
                   'continuous': False}

    def __init__(self, **kwargs):
        self._get_2_server_default_param()
        self._set_param(**kwargs)

        self.time_horizon = self._parameters['time_horizon']

        self.expect_reward = []
        self.num_client_time = []
        self.num_client_class = []

    def _get_2_server_default_param(self):
        # System parameters
        self._parameters['num_server'] = 2
        self._parameters['num_class'] = 2
        self._parameters['num_tsk_per_usr'] = 100
        self._parameters['mu'] = np.array([1, 1])
        self._parameters['Lambda'] = 1.2  # total system incoming task rate
        self._parameters['class_prob'] = np.array([0.5, 0.5])  # prior probability of each class of user
        self._parameters['reward_mat'] = np.array([[0.9, 0.1], [0.9, 0.3]])

        # algorithm parameters
        self._parameters['V'] = 21
        self._parameters['gamma'] = 1.1

    def _set_param(self, **kwargs):
        for key in kwargs:
            assert key in self._parameters, "Wrong kwarg {} given".format(key)
            self._parameters[key] = kwargs[key]
            # print('set {} to {}'.format(key, kwargs[key]))

    def run_simulation(self, myopic=False):
        # print(self._parameters)
        Operator = System(self._parameters)
        start_time = time.time()
        # TODO: make number of iterations configurable, and use multi-thread
        for t in range(self.time_horizon):
            Operator.addNewUser()
            if not myopic:
                Operator.update_client_UCB_est()

            Operator.compute_assignProb(myopic)
            Operator.process_task_at_server()
            Operator.incr_time_slot()
            Operator.update_expect_reward()
            self.expect_reward.append(Operator.expected_reward)
            self.num_client_time.append(Operator.num_clients)
            if not self._parameters['continuous']:
                self.num_client_class.append(Operator.num_class_client.copy())
            if t % 1000 == 0:
                print('{}%...'.format(t / self.time_horizon * 100))

        Operator.show_stats()

        print("--- %s seconds ---" % (time.time() - start_time))

    def plot_results(self, save_figure=False):
        """Plot simulation results"""
        fig1, axes = plt.subplots(2, 1, sharex=True)
        self._plot(axes[0], range(0, self.time_horizon), self.expect_reward, xlabel='time', ylabel='expected reward')
        self._plot(axes[1], range(0, self.time_horizon), self.num_client_time, xlabel='time', ylabel='number of total clients')

        if not self._parameters['continuous']:
            fig2 = plt.figure()
            ax = fig2.add_subplot(1, 1, 1)
            self._plot(ax, range(0, self.time_horizon), self.num_client_class, 'time', 'number of clients per class',
                       None, (['Class {}'.format(i) for i in range(self._parameters['num_class'])]))
        if save_figure:
            figure_dir = 'Figures/'
            if not os.path.exists(figure_dir):
                os.mkdir(figure_dir)
            fig1.savefig(figure_dir+'ExpectedReward_time_classNum_'+str(self._parameters['num_class'])+'.png', bbox_inches='tight')
            fig2.savefig(figure_dir+'NumberClients_time_classNum_'+str(self._parameters['num_class'])+'.png', bbox_inches='tight')

    def _plot(self, ax, x_data, y_data, xlabel=None, ylabel=None, title=None, _legend=None):
        """ helper function for plotting"""
        ax.plot(x_data, y_data)
        ax.grid(which='major', axis='y', alpha=0.3)
        props = {'title': title, 'xlabel': xlabel, 'ylabel': ylabel}
        ax.set(**props)
        if _legend:
            plt.gca().legend(_legend)


def cmd_parser():
    parser = argparse.ArgumentParser(prog='QueueLearning', description='Run queueing with learning simulations.')
    parser.add_argument('--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="insert output filename")

    parser.add_argument("-s", "--save", action='store_true', help="save simulation results/plots")
    parser.add_argument("-p", "--plot", action='store_true', help="show plots")
    parser.add_argument("-v", "--verbose", action='store_true', help="show real-time simulation details")
    parser.add_argument("--version", action='version', version='%(prog)s 1.0')
    parser.add_argument('-cont', '--continuous', action='store_true',
                        help='set payoff vectors to be continuous distributed (default: discrete)')
    parser.add_argument('-m', '--myopic', action='store_true',
                        help='use myopic matching strategy')
    parser.add_argument('-time', '--time_horizon', type=int, default=10000, help="time horizon for each simulation")
    parser.add_argument('-iter', '--iteration', type=int, default=1, help="number of total independent runs")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cmd_parser()

    expect_reward_alg1_all = []

    for iteration in range(args.iteration):
        q1 = queueLearn_time(time_horizon=args.time_horizon, continuous=args.continuous, verbose=args.verbose)
        q1.run_simulation(myopic=False)
        expect_reward_alg1_all.append(q1.expect_reward)

    alg1_expect_reward_mean = np.mean(expect_reward_alg1_all, axis=0)

    if args.save:
        np.savetxt("expected_reward_alg1.txt", alg1_expect_reward_mean, delimiter=",", fmt="%.3f")
        if args.plot:
            q1.plot_results(args.save)

    # for myopic matching simulation
    if args.myopic:
        expect_reward_myopic_all = []
        for iteration in range(args.iteration):
            q2 = queueLearn_time(time_horizon=args.time_horizon, continuous=args.continuous, verbose=args.verbose)
            q2.run_simulation(myopic=True)
            expect_reward_myopic_all.append(q2.expect_reward)
        myopic_expect_reward_mean = np.mean(expect_reward_myopic_all, axis=0)

        if args.save:
            np.savetxt("expected_reward_myopic.txt", myopic_expect_reward_mean, delimiter=",", fmt="%.3f")
            if args.plot:
                q2.plot_results(args.save)

    plt.show()

