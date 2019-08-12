import numpy as np
from numpy import linalg as LA
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt

""" 
KKT_sol is a Fast solution derived from KKT condition if we only have 2 servers
This can solve the solution at least 3x faster comparing to the general solver minimize_solver_sol
For server size > 2, the general solver minimize_solver_sol will be used.
"""


def solve_one_user_KKT(ucb_sorted, V, Mu):
    q = np.random.rand(2)
    q_prev = np.random.rand(2)
    epsilon = np.power(10.0, -6)
    p_opt_kkt = Mu
    step_size = 0.1
    ucb_sorted = ucb_sorted[0]
    while LA.norm(q-q_prev) > epsilon:
        if ucb_sorted[0] - ucb_sorted[1] > q[0] - q[1]:  # send to server 0
            p_opt_kkt = np.array([1/V/(q[0] - ucb_sorted[0]), 0.0])
        elif ucb_sorted[0] - ucb_sorted[1] < q[0] - q[1]:  # send to server 1
            p_opt_kkt = np.array([0, 1/V/(q[1] - ucb_sorted[1])])
        else:  # send to both
            p_opt_kkt = np.array([q_prev[1]/(np.sum(q_prev) + epsilon), q_prev[0]/(np.sum(q_prev) + epsilon)]) * np.sum(p_opt_kkt)
            p_opt_kkt = np.sum(p_opt_kkt) / 2.0 * np.ones(2)

        q_prev = q.copy()
        q[0] = np.maximum(q_prev[0] + step_size * (p_opt_kkt[0] - Mu[0]), 0)
        q[1] = np.maximum(q_prev[1] + step_size * (p_opt_kkt[1] - Mu[1]), 0)

    return p_opt_kkt


def KKT_sol(ucb_est, mu, V):
    """
    ucb_est is a numpy array with shape (n_client, n_server = 2)
    the entries of ucb_est should be < 0

    """
    if ucb_est.shape[1] != 2:
        raise(ValueError, "This solution is only valid for server size = 2")

    if ucb_est.shape[0] == 0:
        return np.zeros((0, 2))

    ucb_diff = ucb_est[:, 0] - ucb_est[:, 1]

    n = ucb_est.shape[0]  # number of clients
    # sorting in descending order
    sort_idx = (-ucb_diff).argsort()
    ucb_sorted = ucb_est[sort_idx, :]
    p_opt_kkt = np.zeros((n, 2))

    if n == 1:
        p_opt_kkt[0, :] = solve_one_user_KKT(ucb_sorted, V, mu)
        return p_opt_kkt

    q = np.array([0.0, 0.0])

    def func(qq, *params):
        indices, server_j = params
        return np.sum(1 / (qq - ucb_sorted[indices, server_j])) - V * mu[server_j]

    def func2(qq, *params):
        k_id, indices_1, indices_2 = params
        return np.sum(1 / (qq - ucb_sorted[indices_1, :])) \
               + np.sum(1 / (qq - ucb_sorted[k_id, 0] + ucb_sorted[k_id, 1] - ucb_sorted[indices_2, 1])) \
               - V * np.sum(mu)

    def solve_q(q_prev, I, server_j):
        q_sol = fsolve(func, q_prev+0.5, args=(I, server_j))
        q_sol = np.maximum(q_sol, 0)
        return q_sol

    # start searching in the middle for speed optimization
    # First test I1 = [0:m-1] and I2 = [m:n-1]
    m = int(n/2)
    I1 = list(range(0, m))
    I2 = list(range(m, n))
    q[0] = solve_q(0.5, I1, 0)
    q[1] = solve_q(0.5, I2, 1)
    global k

    # index = min(m, n-1)
    k = m

    if ucb_sorted[m, 0] - ucb_sorted[m, 1] > q[0] - q[1]:   # should assign m to server 0
        for k in range(m, n):
            I1 = list(range(0, k+1))
            I2 = list(range(k+1, n))
            q[0] = solve_q(q[0], I1, 0)
            q[1] = 0 if not I2 else solve_q(q[1], I2, 1)

            if ucb_sorted[min(k+1,n-1), 0] - ucb_sorted[min(k+1, n-1), 1] > q[0] - q[1]: # should assign more to server 0
                continue
            elif ucb_sorted[k, 0] - ucb_sorted[k, 1] < q[0] - q[1]:  # share traffic for client k between 2 servers
                # solve the two equations for q1, q2
                param = (k, I1, I2)
                q[0] = fsolve(func2, q[0], args=param)
                q[1] = q[0] - ucb_sorted[k, 0] + ucb_sorted[max(k-1, 0), 1]
                if min(q) < 0:
                    q -= min(q)
                break
            else:
                break
    elif ucb_sorted[max(m-1, 0), 0] - ucb_sorted[max(m-1, 0), 1] < q[0] - q[1]:  # should assign m to server 1
        for k in range(m-2, -1, -1):  # find index downward until 0
            I1 = list(range(0, k+1))
            I2 = list(range(k+1, n))

            # update dual variables
            q[0] = 0 if not I1 else solve_q(q[0], I1, 0)
            q[1] = solve_q(q[1], I2, 1)

            if ucb_sorted[k, 0] - ucb_sorted[k, 1] < q[0] - q[1]:  # should assign k to server 1
                continue
            elif ucb_sorted[min(k+1, n-1), 0] - ucb_sorted[min(k+1, n-1), 1] > q[0] - q[1]:
                param = (k+1, I1, I2)
                q[0] = fsolve(func2, q[0], args=param)
                q[1] = q[0] - ucb_sorted[k, 0] + ucb_sorted[k, 1]
                if min(q) < 0:
                    q -= min(q)
                break
            else:
                break

    p_opt_kkt[I1, 0] = 1 / V / (q[0] - ucb_sorted[I1, 0])
    p_opt_kkt[I2, 1] = 1 / V / (q[1] - ucb_sorted[I2, 1])

    # spread exceeded-traffic from server 1 to server 2
    delta0 = np.sum(p_opt_kkt[:, 0]) - mu[0]
    if delta0 > 0:
        p_opt_kkt[max(k-1, 0), 0] -= delta0
        p_opt_kkt[max(k-1, 0), 1] += delta0

    delta1 = np.sum(p_opt_kkt[:, 1]) - mu[1]
    if delta1 > 0:  # user k splitting traffic
        p_opt_kkt[min(k, n-1), 0] += delta1
        p_opt_kkt[min(k, n-1), 1] -= delta1

    id_rev = np.argsort(sort_idx)
    p_opt_kkt_return = p_opt_kkt[id_rev, :]

    return p_opt_kkt_return


def minimize_solver_sol(ucb_m_gamma, mu, V, verbose=False):
    """Compute client to server probability assignment"""

    if ucb_m_gamma.shape[0] == 0:
        return np.zeros((0, 2))

    num_client = ucb_m_gamma.shape[0]
    num_server = ucb_m_gamma.shape[1]
    p_client_server = np.ones((num_client, num_server))

    xinit = np.ones(num_client * num_server)   # important to initialize from large values

    A = np.zeros((num_server, num_server * num_client))
    for j in range(num_server):
        A[j, j: (num_server * num_client): num_server] = 1

    func_val = []

    def obj_dynamic(x):
        f = 0.0
        epsilon = np.power(10.0, -4.0)
        for i in range(num_client):
            prob_to_server_sum = np.sum(x[i*num_server: (i+1)*num_server])
            temp_sum = x[i*num_server: (i+1)*num_server].dot(ucb_m_gamma[i, :])

            f += 1/V * np.log(prob_to_server_sum + epsilon) + temp_sum  # add eps to avoid log(0)

        func_val.append(-f)

        return -f

    def ineq_const(x):
        return mu - A @ x

    """ Reference:
     1. https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html  (method='SLSQP')
     2. https://stackoverflow.com/questions/52001922/linearconstraint-in-scipy-optimize/52003654
    """
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
        plt.title('Alg-1 obj.')
        plt.show()

    # return number of expected tasks sent
    return p_client_server

    """Using 'trust-constr' solver not work well"""
        # lb = np.zeros(num_server * num_client)
        # ub = np.ones(num_server * num_client)
        # bounds = Bounds(lb, ub)
        # linear_constraint = LinearConstraint(A, np.zeros(2), b)
        # res = minimize(obj_dynamic, x0=xinit, method='trust-constr',
        #                options={'verbose': 1},
        #                constraints=linear_constraint,
        #                bounds=bounds)

def myopic_matching_solver(payoffs, mu_vec, verbose=False):
    """Compute client to server myopic probability assignment"""

    if payoffs.shape[0] == 0:
        return np.zeros((0, 2))

    num_client = payoffs.shape[0]
    num_server = payoffs.shape[1]
    p_client_server = np.ones((num_client, num_server))

    xinit = np.ones(num_client * num_server) / (num_client * num_server)  # important to initialize from large values

    A = np.zeros((num_server, num_server * num_client))
    for j in range(num_server):
        A[j, j: (num_server * num_client): num_server] = 1

    func_val = []

    def obj_myopic(x):
        f = 0.0
        for l in range(num_client):
            temp_sum = x[l*num_server: (l+1)*num_server].dot(payoffs[l, :])
            f += temp_sum

        f_val = -f
        func_val.append(f_val)

        return f_val

    # server constraint
    def ineq_const(x):
        return mu_vec - A @ x

    # probability constraint
    # def eq_const(x):
    #     return np.ones(num_client) - np.array([np.sum(x[l*num_server: (l+1)*num_server]) for l in range(num_client)])

    # eq_cons = {'type': 'eq',
    #            'fun': eq_const}

    ineq_cons = {'type': 'ineq',
                 'fun': ineq_const}

    # probability
    # bds = [(0, 1) for _ in range(num_client * num_server)]
    bds = [(0, mu_vec[j]) for _ in range(num_client) for j in range(num_server)]

    res = minimize(obj_myopic, x0=xinit, method='SLSQP',
                   options={'disp': verbose},
                   constraints=ineq_cons,
                   bounds=bds)

    if res.success:
        p_opt = res.x
    else:
        print(res)
        raise(TypeError, "Cannot find a valid solution by SLSQP")

    for i in range(num_client):
        p_client_server[i, :] = p_opt[i*num_server: (i+1)*num_server]

    if verbose:
        plt.plot(func_val)
        plt.title('Myopic obj.')
        plt.show()

    return p_client_server


