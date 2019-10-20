# QueueLearning
This repository provides a client-server matching solution to maximize system payoffs for online service platform settings. It is useful for early stage platforms as a practical solution to match the "client" to the most suitable "server" when the knowledge to clients are very limited.

## Introduction
We assume the service platform initially has limited information for new coming clients but can learn from client-server generated payoffs as feedback to improve its matching performance. The only information that the service platform would need is the (average) service rate for each server. In particular, the proposed solution incorporates online learning and adaptive queueing control so that the solution applies to situations with following unknowns:
1. Unknown client arrival rates
2. Unknown client/server-dependent payoffs
3. Unknown number of tasks/trials from clients
4. Uncertain (random) service rate at servers.

The algorithm nicely balances the exploration-and-exploitation tradeoff while ensuring system stability to achieve much better system payoff than myopic optimal matching.

## Usage of codes
In Algorithm1.py, we provide a working example for the proposed algorithm, Algorithm 1.

The command to run Algorithm1.py is as follows:
`python3 Algorithm1.py`

This example assumes a 2-server system with service rate of 1 for both servers.
The Algorithm 1 solver takes in 4 parameters:
1. Reward matrix. Client-server upper-confidence-bound (UCB) payoff matrix.
2. Service rate. A vector consists of servers' service rate (number of tasks/time-slot).
3. V. A system parameter. Controlling the conservativeness of task assignment.
4. Gamma. A system parameter. Controlling the conservativeness of task assignment.

Increasing parameters V and Gamma lead to a better client-server payoff estimates with the cost of a larger queue in the system.

---
Another main_queueLearn.py is a more complete script simulating the system dynamic using Algorithm 1. The new client is assumed to join the system as a Bernoulli process, and
the task assignment is computed with Algorithm 1 solution. We track the number of clients (based on their underlying class) and the expected system payoff for each time-slot.

To use the default setting of 2-class 2-server simulation, we can run
`python3 main_queueLearn.py`

The user can do customized simulation by specifying args parameters such as,
`python3 main_queueLearn.py -t 25000 -o expected_reward_alg1.txt -p -s`

to simulate a system dynamics with time horizon 25000, save the expected system reward  to output file "expected_reward_alg1.txt" and the plotted figures.
For more detailed descriptions of the code usage, please do
`python3 main_queueLearn.py -h`


## Contributors
The system framework and Algorithm 1 is developed and analyzed by Prof. Xiaojun Lin, Prof. Jiaming Xu and Wei-Kang Hsu. The example code is maintained by Wei-Kang Hsu.
