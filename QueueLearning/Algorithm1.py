import numpy as np
import argparse
from numpy import linalg as LA
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt


def Algorithm1(ucb_m_gamma, mu, V, verbose=False):
    """Compute client to server probability assignment"""

    if ucb_m_gamma.shape[0] == 0:
        return np.zeros((0, 2))

    (num_client, num_server) = ucb_m_gamma.shape
    p_client_server = np.ones((num_client, num_server))

    xinit = np.ones(num_client * num_server) * max(mu)  # important to initialize to larger values

    A = np.zeros((num_server, num_server * num_client))

    for j in range(num_server):
        A[j, j: (num_server * num_client): num_server] = 1

    func_val = []

    def obj_dynamic(x):
        """The objective function to minimize"""
        f = 0.0
        epsilon = np.power(10.0, -6.0)
        for i in range(num_client):
            prob_to_server_sum = np.sum(x[i*num_server: (i+1)*num_server])
            temp_sum = x[i*num_server: (i+1)*num_server].dot(ucb_m_gamma[i, :])

            f += 1/V * np.log(prob_to_server_sum + epsilon) + temp_sum  # add eps to avoid log(0)

        func_val.append(-f)
        return -f

    def ineq_const(x):
        return mu - A @ x

    ineq_cons = {'type': 'ineq',
                 'fun': ineq_const}

    bds = [(0, mu[j]) for _ in range(num_client) for j in range(num_server)]

    res = minimize(obj_dynamic, x0=xinit, method='SLSQP',
                   options={'disp': verbose},
                   constraints=ineq_cons,
                   bounds=bds)

    if res.success:
        p_opt = res.x
    else:
        raise(TypeError, "Cannot find a valid solution by SLSQP")

    for i in range(num_client):
        p_client_server[i, :] = p_opt[i*num_server: (i+1)*num_server]

    if verbose:
        plt.plot(func_val)
        plt.title('Alg-1 obj. function')
        plt.xlabel('Number of iterations')
        plt.show()

    # return number of expected tasks sent
    return p_client_server


def cmd_parser():
    parser = argparse.ArgumentParser(prog='Algorithm 1', description='Minimal working example to run Algorithm 1')
    parser.add_argument('-v', '--verbose', action='store_true', help='display detail information of Algorithm 1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = cmd_parser()

    # System parameters
    """ Control the conservativeness of the system, i.e., the larger the value, the longer time to explore and assign"""
    Gamma = 1.1  # the value have to be larger than (>) maximum value of ucb_reward_matrix defined below.
    V = 20

    # User specified parameters
    """ The UCB estimated reward matrix.
    Each row contains a client's estimate for how good to assign task to each server.
    Note that, the matrix is not the same as true payoff estimates.
    Instead, it is be the truncated ucb estimates for each client to every server.
    E.g., client-1 has ucb estimates of 0.9 and 0.3 for server 1 and 2, respectively.
    But client-1's true payoff estimate for the two servers could be 0.6 and 0.2.
    """
    ucb_reward_matrix = np.array([[0.9, 0.3], [0.9, 0.1]])

    """Assume two servers, each with service rate 1 task/time-slot"""
    service_rate = np.array([1, 1])

    # Compute assign probability
    computed_assign_task = Algorithm1(ucb_reward_matrix - Gamma, service_rate, V, verbose=args.verbose)

    print("UCB reward matrix\n{}\n".format(ucb_reward_matrix))
    print("Service rate\n{}\n".format(service_rate))
    print("Task-assignment matrix \n{}\n".format(computed_assign_task))
