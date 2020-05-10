import time
from collections import deque
from random import random

import matplotlib.pyplot as plt
import numpy as np
import yaml

from maddpg.parameters import ParReacher


class Plotting:
    def __init__(self, args):
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.scores_per_agent = []
        self.scores_std = []
        self.scores_mean = []
        self.lower = []
        self.upper = []
        self.scores_min = []
        self.scores_max = []
        self.epsilon = []

        self.count = 0
        self.score_int = 0
        self.score_ints_x = []
        self.score_ints = []
        self.eps_next = 1
        self.eps_x = []
        self.eps_y = []

        self.scores_mean_max = -np.inf
        self.scores_mean_max_idx = 0
        self.fname = "{}".format(time.time())

    def add_measurement(self, score_per_agent, eps):
        score = np.mean(score_per_agent)
        self.scores_per_agent.append(score_per_agent)
        self.scores.append(np.mean(score))
        self.scores_window.append(score)
        self.epsilon.append(eps)
        if eps <= self.eps_next:
            self.eps_next *= 0.5
            self.eps_x.append(self.count)
            self.eps_y.append(eps)
        mean_score = np.mean(self.scores_window)
        if mean_score >= self.score_int + 1:
            self.score_int += 1
            self.score_ints_x.append(self.count)
            self.score_ints.append(self.score_int)
        if mean_score > self.scores_mean_max:
            self.scores_mean_max = mean_score
            self.scores_mean_max_idx = self.count

        self.scores_std.append(np.std(self.scores_per_agent[-1]))

        # the score for "solving" the requirement is the mean of all agents over the last 100 episodes
        self.scores_mean.append(mean_score)
        self.lower.append(score - self.scores_std[-1])
        self.upper.append(score + self.scores_std[-1])

        self.count += 1

    # do some logging and plotting
    def plotting(self, args):
        # plot the scores
        # fig = plt.figure(num=id)
        id = 1
        fig, axs = plt.plot(1, constrained_layout=True, num=id)
        # score for each agent
        axs[0].plot(np.arange(len(self.scores)), self.scores_per_agent, linestyle='-.', linewidth=0.3)
        axs[0].plot(np.arange(len(self.scores)), self.scores, label='score')
        axs[0].plot(np.arange(len(self.scores_mean)), self.scores_mean, label='100 mean score')
        axs[0].plot(self.score_ints_x, self.score_ints, '.', label='mean score int')
        axs[0].plot(np.arange(len(self.lower)), self.lower, label='upper sigma-confidence')
        axs[0].plot(np.arange(len(self.upper)), self.upper, label='lower sigma-confidence')

        axs[0].legend()
        axs[0].set_ylabel('Score')

        # axs[1].plot(np.arange(len(self.epsilon)), self.epsilon, label='epsilon')
        # axs[1].set_xlabel('Episode Number')
        plt.savefig(self.fname + '.png')
        plt.close(id)
        self._write_yaml(args)

    def _write_yaml(self, args):
        dict_file = [
            {'max_score': float(self.scores_mean_max)},
            {'at_iteration': int(self.scores_mean_max_idx)},
            {'buffer_size': args.buffer_size},
            {'batch_size': args.batch_size},
            {'gamma': args.gamma},
            {'tau': args.tau},
            {'lr_actor': args.lr_actor},
            {'lr_critic': args.lr_critic},
            {'weight_decay': args.weight_decay},
            {'ou_mu': args.ou_mu},
            {'ou_theta': args.ou_theta},
            {'ou_sigma': args.ou_sigma},
            {'actor_fc1_units': args.actor_fc1_units},
            {'actor_fc2_units': args.actor_fc2_units},
            {'critic_fcs1_units': args.critic_fcs1_units},
            {'critic_fc2_units': args.critic_fc2_units},
            {'random_seed': args.random_seed},
            {'update_every': args.update_every},
            {'num_updates': args.num_updates},
            {'epsilon': args.epsilon},
            {'epsilon_decay': args.epsilon_decay},
            {'num_episodes': args.num_episodes}]

        yaml_name = self.fname + '.yaml'
        with open(yaml_name, 'w+') as yaml_file:
            yaml.dump_all([dict_file, list(self.scores_mean), list(self.scores), list(self.epsilon)],
                          yaml_file)


# main function
if __name__ == "__main__":
    args = ParReacher()

    plotting = Plotting(args)
    for i in range(100):
        scores = [random() * 10 + min(10, 1.1 * i) for x in range(20)]
        epsilon = 1 / (i + 1)
        plotting.add_measurement(score_per_agent=scores, eps=epsilon)
    start_time = time.time()
    plotting.plotting(args=args)
    runtime = time.time() - start_time
    print('runtime is {:.2f}'.format(runtime))
