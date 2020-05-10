import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from maddpg.ddpg_model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, par):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            par (Par): parameter object
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(par.random_seed)
        self.epsilon = par.epsilon

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, par).to(device)
        self.actor_target = Actor(state_size, action_size, par).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=par.lr_actor)
        print('actor')
        print(self.actor_local)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, par).to(device)
        self.critic_target = Critic(state_size, action_size, par).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=par.lr_critic,
                                           weight_decay=par.weight_decay)
        print('critic')
        print(self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, par.random_seed, par.ou_mu, par.ou_theta, par.ou_sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, par.buffer_size, par.batch_size, par.random_seed)

        # Make sure target is with the same weight as the source
        # The seed makes sure the networks are the same
        # self.hard_copy(self.actor_target, self.actor_local)
        # self.hard_copy(self.critic_target, self.critic_local)

        self.time_learn = deque(maxlen=100)
        self.time_act = deque(maxlen=100)
        self.epsilon = 1

        self.par = par

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """
        Save experience in replay memory and use random sample from buffer to learn.
        :param states: state of the environment
        :param actions: executed action
        :param rewards: observed reward
        :param next_states: subsequent state
        :param dones: boolean signal indicating a finished episode
        :return:
        """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.par.batch_size and timestep % self.par.update_every == 0:
            for _ in range(self.par.num_updates):
                experiences = self.memory.sample()
                self.learn(experiences, self.par.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()
        # clipping is done with tanh output layers
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.par.tau)
        self.soft_update(self.actor_local, self.actor_target, self.par.tau)

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= self.par.epsilon_decay
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """
        soft update model parameters
        :param local_model: PyTorch model (weights will be copied from)
        :param target_model: PyTorch model (weights will be copied to)
        :param tau: interpolation parameter
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.05, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a replay buffer object
        :param action_size:
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param seed:
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory and return it on device """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
