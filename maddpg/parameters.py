class Par:
    def __init__(self):
        # Parameters suggested according to this paper: Continuous control with deep reinforcement learning
        # https://arxiv.org/pdf/1509.02971.pdf

        # Learning hyperparameters
        self.buffer_size = int(1e6)  # replay buffer size
        self.batch_size = 64  # minibatch size
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        self.lr_actor = 1e-4  # learning rate of the actor
        self.lr_critic = 1e-3  # learning rate of the critic
        self.weight_decay = 1e-2  # L2 weight decay

        # ou noise
        self.ou_mu = 0.
        self.ou_theta = 0.15
        self.ou_sigma = 0.25

        # network architecture for actor and critic
        self.actor_fc1_units = 400
        self.actor_fc2_units = 300
        self.critic_fcs1_units = 400
        self.critic_fc2_units = 300

        # Further parameter not found in paper
        self.random_seed = 15  # random seed
        self.update_every = 1  # timesteps between updates
        self.num_updates = 1  # num of update passes when updating
        self.epsilon = 1.0  # epsilon for the noise process added to the actions
        self.epsilon_decay = 1e-6  # decay for epsilon above
        self.num_episodes = 1000  # number of episodes


class ParReacher(Par):
    def __init__(self):
        super(ParReacher, self).__init__()

        # tuned parameter to "reach" the goal

        # Learning
        self.batch_size = 256  # minibatch size
        self.lr_actor = 1e-3  # learning rate of the actor
        self.weight_decay = 0  # L2 weight decay

        # ou noise
        self.ou_theta = 0.15
        self.ou_sigma = 0.05

        # network architecture for actor and critic
        self.actor_fc1_units = 128
        self.actor_fc2_units = 128
        self.critic_fcs1_units = 128
        self.critic_fc2_units = 128


class ParCrawler(Par):
    def __init__(self):
        super(ParCrawler, self).__init__()
