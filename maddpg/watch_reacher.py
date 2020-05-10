import numpy as np
import torch
from unityagents import UnityEnvironment

from maddpg.ddpg_agent import Agent
# Get Reacher Environment
from maddpg.parameters import ParReacher

par = ParReacher()
env = UnityEnvironment(file_name=par.file_name_watch)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment and set rendering to real time (train mode is false)
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# setup and load agent
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]

agent = Agent(state_size=state_size, action_size=action_size, par=par)
agent.actor_local.load_state_dict(torch.load('best_checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('best_checkpoint_critic.pth'))

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
states = env_info.vector_observations  # get the current state (for each agent)
scores = np.zeros(num_agents)  # initialize the score (for each agent)
while True:
    actions = agent.act(states)  # select an action (for each agent)
    env_info = env.step(actions)[brain_name]  # send all actions to tne environment
    next_states = env_info.vector_observations  # get next state (for each agent)
    rewards = env_info.rewards  # get reward (for each agent)
    dones = env_info.local_done  # see if episode finished
    scores += env_info.rewards  # update the score (for each agent)
    states = next_states  # roll over states to next time step
    if np.any(dones):  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
