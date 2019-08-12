import numpy as np
from assign_alg import KKT_sol
from assign_alg import minimize_solver_sol
from assign_alg import myopic_matching_solver

np.random.seed(0)
# ucb_est = np.array(np.random.rand(5, 3))

# Test 1
# ucb_est = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [1.0, 0.7], [0.2, 0.8]])

ucb_est = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.9, 0.5], [1.0, 0.5]])

# Test 2
# ucb_est = np.zeros((10, 2))
# ucb_est[:3, 1] = 0.5
# ucb_est[3:7, 0] = 0.5
# ucb_est[7:, 1] = 1

# Test 3
# ucb_est = np.array([[1.0, 1.0], [1.0, 1.0]])

V = 20
mu = np.array([1.0, 1.0])
gamma = 1.1

assert ucb_est.shape[1] == len(mu), 'Wrong number of servers'

ucb_adjusted = ucb_est - gamma

print("adjusted UCB")
print(ucb_adjusted)

prob_assign = KKT_sol(ucb_adjusted, mu, V)
print("KKT prob_assign : ")
print(prob_assign)
print("Sum prob.")
print(np.sum(prob_assign, axis=0))

prob_assign_fmin = minimize_solver_sol(ucb_adjusted, mu, V, verbose=False)

print('----------------------------')
print("fmincon prob_assign: ")
print(prob_assign_fmin)
print("Sum prob.")
print(np.sum(prob_assign_fmin, axis=0))


prob_assign_mypoic = myopic_matching_solver(ucb_est, mu, verbose=True)
print('----------------------------')
print("myopic fmincon prob_assign: ")
print(prob_assign_mypoic)
print("Sum prob.")
print(np.sum(prob_assign_mypoic, axis=0))

