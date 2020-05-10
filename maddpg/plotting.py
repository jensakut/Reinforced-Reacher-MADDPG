import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    def __init__(self):
        self.scores = []
        self.scores_mean = []
        self.lower = []
        self.upper = []
        self.scores_min = []
        self.scores_max = []
        self.episode_length = []
        self.epsilon = []

    def add_measurement(self, score, episode_length, epsilon):
        self.scores.append(score)
        self.episode_length.append(episode_length)
        self.epsilon.append(epsilon)

        l = min(100, len(self.scores))
        self.scores_mean.append(np.mean(self.scores[-l:]))
        # self.scores_min.append(np.min(self.scores[-l:]))
        # self.scores_max.append(np.max(self.scores[-l:]))
        std = np.std(self.scores[-l:])
        mean = self.scores_mean[-1]
        self.lower.append(mean - std)
        self.upper.append(mean + std)

    # do some logging and plotting
    def plotting(self, name, id):
        # plot the scores
        # fig = plt.figure(num=id)
        fig, axs = plt.subplots(3, 1, constrained_layout=True, num=id, dpi=500)
        axs[0].plot(np.arange(len(self.scores)), self.scores, label='score')
        axs[0].plot(np.arange(len(self.scores_mean)), self.scores_mean, label='100 mean score')
        axs[0].plot(np.arange(len(self.lower)), self.lower, label='+std')
        axs[0].plot(np.arange(len(self.upper)), self.upper, label='-std')
        # axs[0].plot(np.arange(len(self.scores_min)), self.scores_min, label='100 min score')
        # axs[0].plot(np.arange(len(self.scores_max)), self.scores_max, label='100 max score')
        axs[0].legend()
        axs[0].set_ylabel('Score')

        axs[1].plot(np.arange(len(self.episode_length)), self.episode_length, label='episode_length')
        axs[1].set_ylabel('steps per episode')

        axs[2].plot(np.arange(len(self.epsilon)), self.epsilon, label='epsilon')
        axs[2].set_ylabel('epsilon')

        axs[2].set_xlabel('Episode Number')
        plt.savefig(name)
        plt.close(id)
