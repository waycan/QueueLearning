import numpy as np
from Client import Client
from assign_alg import KKT_sol, minimize_solver_sol, myopic_matching_solver


class System(object):

    def __init__(self, sys_param):
        self.continuous_class = sys_param['continuous']
        self.clients = {}
        self.idx_to_client_mapping = {}
        self.client_tag = 0
        self.num_clients = 0
        self.total_reward = 0
        self.expected_reward = 0
        self.time_slot = 0
        self.assign_prob = 0
        self.V = sys_param['V']
        self.gamma = sys_param['gamma']
        self.mu = sys_param['mu']
        self.num_server = sys_param['num_server']
        self.reward_matrix = sys_param['reward_mat']
        self.num_avg_task = sys_param['num_tsk_per_usr']
        self.ucb_est_clients = np.zeros((self.num_clients, sys_param['num_server']))
        self.time_horizon = sys_param['time_horizon']
        self.user_in_rate = np.random.rand(2, self.time_horizon)

        if sys_param['continuous']:
            self.Lambda = sys_param['Lambda']
            # raise NotImplementedError
        else:
            self.lambda_i = (sys_param['Lambda'] * sys_param['class_prob']) / sys_param['num_tsk_per_usr']  # average incoming rate per user class
            self.num_class_client = np.zeros(len(sys_param['class_prob']))
            self.class_server_assignTasks = np.zeros((len(sys_param['class_prob']), sys_param['num_server']))
            self.num_total_class_client = np.zeros(len(sys_param['class_prob']))

        self.verbose = sys_param['verbose']
        self.num_client_in = 0
        self.num_client_leave = 0


    def addNewUser(self):
        """ the client comes into the system as a Bernoulli(p) process """
        if self.continuous_class:
            if np.random.rand() < self.Lambda/self.num_avg_task:
                num_task = np.random.geometric(p=1/self.num_avg_task)
                reward_vec = np.random.rand(2)
                #  assume server 1 has the higher payoff among the two
                reward_vec = reward_vec if reward_vec[0] > reward_vec[1] else np.flip(reward_vec)
                self.clients[self.client_tag] = Client(self.client_tag, num_task, self.num_server, reward_vec)
                self.client_tag += 1
                self.num_clients += 1
                self.num_client_in += 1

                if self.verbose:
                    print("Add (client, # task, time) =  ({}, {}, {}) ".format(self.client_tag, num_task, self.time_slot))
        else:
            for c_label, rate in enumerate(self.lambda_i):
                # if np.random.rand() < rate:
                if self.user_in_rate[c_label, self.time_slot] < rate:
                    self.num_total_class_client[c_label] += 1
                    num_task = np.random.geometric(p=1/self.num_avg_task)
                    reward_vec = self.reward_matrix[c_label, :]
                    self.clients[self.client_tag] = Client(c_label, num_task, self.num_server, reward_vec)
                    self.client_tag += 1
                    self.num_clients += 1
                    self.num_client_in += 1
                    self.num_class_client[c_label] += 1

                    if self.verbose:
                        print("Add (client, # task, time) =  ({}, {}, {}) ".format(self.client_tag, num_task, self.time_slot))

    def compute_assignProb(self, myopic):
        self.parse_clients()

        if myopic:
            """ myopic matching """
            assign_prob = myopic_matching_solver(self.ucb_est_clients, self.mu, self.verbose)
        # run Algorithm 1 below
        elif self.num_server == 2:
            """ KKT solution """
            assign_prob = KKT_sol((self.ucb_est_clients - self.gamma), self.mu, self.V)
        else:
            """ solve optimization problem """
            assign_prob = minimize_solver_sol((self.ucb_est_clients - self.gamma), self.mu, self.V, self.verbose)

        self.assign_prob = assign_prob

    def getClientId(self, client_to_serve_idx):
        """
            input: relative client index in the system
            output: client tag_id
        """
        return self.idx_to_client_mapping[client_to_serve_idx]

    def get_client_UCB_est(self, c_id):
        """
            :param c_id: true client tag_id
            :return:  ucb estimates for all servers from the client
        """
        return self.clients[c_id].ucb_est

    def reset(self):
        """
            reset clients info every frame before parsing clients
        """
        self.ucb_est_clients = np.zeros((self.num_clients, self.num_server))
        self.idx_to_client_mapping = {}

    def parse_clients(self):
        """
            fill in the clients information needed for system to compute assign probability
        """

        self.reset()

        for system_id, client_tag in enumerate(self.clients.keys()):
            self.idx_to_client_mapping[system_id] = client_tag
            self.ucb_est_clients[system_id, :] = self.get_client_UCB_est(client_tag)

        return

    def process_task_at_server(self):
        client_to_remove = set()  # save the client tag to be removed at the end of time_slot

        for j in range(self.num_server):
            for nu in range(self.mu[j]):
                cum_prob = np.cumsum(self.assign_prob[:, j] / self.mu[j])
                client_to_serve_idx = self.find_first(np.random.rand(), cum_prob)

                if client_to_serve_idx != -1:  # found client to server
                    client_tag = self.getClientId(client_to_serve_idx)
                    client_obj = self.clients[client_tag]

                    if client_obj.hasNoTask():
                        reward = 0
                        client_to_remove.add(client_tag)
                    else:
                        reward = client_obj.generate_reward(j)
                        client_obj.update_remainTask()
                        if not self.continuous_class:
                            self.class_server_assignTasks[client_obj.class_label, j] += 1

                    if self.verbose:
                        print("Match: (time, client, server, task_remain) = ({}, {}, {}, {})".format(self.time_slot,
                                                                                                 client_tag, j,
                                                                                                 client_obj.num_remain_tsk))
                    self.total_reward += reward

        self.update_users_system(client_to_remove)

    def update_users_system(self, client_to_remove):
        self.remove_done_clients(client_to_remove)
        self.num_clients -= len(client_to_remove)
        self.num_client_leave += len(client_to_remove)

    def update_client_UCB_est(self):
        for c in self.clients.values():
            c.computeTrunUCB()

    def remove_done_clients(self, client_to_remove):
        for k in client_to_remove:
            if not self.continuous_class:
                self.num_class_client[self.clients[k].class_label] -= 1
            self.clients.pop(k, None)
            if self.verbose:
                print("Client leaving: {}".format(k))

    def incr_time_slot(self):
        self.time_slot += 1

    def update_expect_reward(self):
        self.expected_reward = self.total_reward / self.time_slot

    """utilities"""
    def find_first(self,item, vec):
        """return the index of the first occurence of item in vec"""
        for i, cum_p in enumerate(vec):
            if item <= cum_p:
                return i
        return -1

    def show_stats(self):
        print("Simulation horizon: {}".format(self.time_slot))
        print("Number of total clients in: {}".format(self.num_client_in))
        print("Number of total clients leave: {}".format(self.num_client_leave))
        print("Total / expected reward = {} / {}".format(self.total_reward, self.expected_reward))

        if not self.continuous_class:
            [print(" - Class {} clients in: {}".format(i, int(self.num_total_class_client[i]))) for i in range(2)]
            avg_class_to_server_prob = np.transpose(np.transpose(self.class_server_assignTasks) / self.num_total_class_client)
            print("Average assign probability (per task) = ")
            print(avg_class_to_server_prob / self.num_avg_task)
            print('Average reward (computed) = {}'.format(
                np.sum(self.lambda_i.dot(avg_class_to_server_prob * self.reward_matrix))))




