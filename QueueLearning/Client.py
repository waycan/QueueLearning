import numpy as np


class Client(object):
    def __init__(self, class_label, num_task, num_server, reward_vec):
        assert (reward_vec.shape[0] == num_server), "Reward vector size has to be the same as num_server"
        self.class_label = class_label
        self.num_remain_tsk = num_task
        self.reward_est = np.zeros(num_server)
        self.num_trial_server = np.zeros(num_server)
        self.ucb_est = np.zeros(num_server)
        self.reward_collect_server = np.zeros(num_server)
        self.underlying_reward = reward_vec

    def computeTrunUCB(self):

        for j in range(len(self.ucb_est)):
            if self.num_trial_server[j] > 0:
                self.ucb_est[j] = self.reward_est[j] + \
                                  np.sqrt(2 * np.log(np.sum(self.num_trial_server)) / self.num_trial_server[j])
            else:
                self.ucb_est[j] = 1.0

        # truncated version of UCB
        self.ucb_est = np.minimum(self.ucb_est, 1)

    def hasNoTask(self):
        return self.num_remain_tsk <= 0

    def update_remainTask(self, num_tsk_assign=1):
        self.num_remain_tsk -= num_tsk_assign

    def update_rewardEst(self, r, server_id):
        """r is the reward collected for each task assigned to server server_id"""
        self.num_trial_server[server_id] += 1
        self.reward_collect_server[server_id] += r
        self.reward_est[server_id] = self.reward_collect_server[server_id] / self.num_trial_server[server_id]

    def getClassLabel(self):
        return self.class_label

    def generate_reward(self, server_id):
        """ Assume reward is generated as Bernoulli process """
        reward = int(np.random.rand() <= self.underlying_reward[server_id])
        self.update_rewardEst(reward, server_id)

        return reward


