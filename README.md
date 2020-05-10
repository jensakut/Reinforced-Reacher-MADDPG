[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

#### Udacity Deep Reinforcement Learning Nanodegree 
## Project 2: Continuous Control
# Reinforced Reacher Robot with MADDPG 
### Introduction

This project uses reinforcement learning to control a set of robotic arms. Reinforcement learning is used in many real-world examples. 
This project uses a robotic picker environment to develop and verify multi-agent reinforcement learning in continous control. 

![Alt text](assets/robot-pickers.gif?raw=true "Title")

*Photo credit: [Google AI Blog](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)*


This project is based on the [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that 
the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target 
location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of 
the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the
 action vector should be a number between -1 and 1.

The environment is considered solved if the agent receives an average reward (over 100 episodes) of at least +30, or
the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.


### Distributed Training

- The version contains 20 identical agents, each with its own copy of the environment.  

The version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Getting Started and training the best configuration

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
    
    - Extract the environment into the project folder. 

1. Create (and activate) a new [anaconda](https://www.anaconda.com/distribution/) environment with Python 3.6.

Anaconda takes care of the right cuda installation, as long as a nvidia gpu with the official driver is installed. 
Therfore an installation is well worth it if not already done. Installing cuda manually is ... much more time-consuming 
and difficult.


    - __Linux__ or __Mac__: 
    ```bash
    conda env create -f environment.yml
    conda activate pytorch 
    ```
    - __Windows__: 
    ```bash
    conda create --f environment.yml
    activate pytorch
    ```
2. Install the requirements with 
    ```bash
    pip install .
    ```

3. Train the agent or watch a smart agent with the following scripts
    ```bash
    python maddpg/train.py
    python maddpg/watch_reacher.py
    ```



## Results: 

Using the parameters from the [ddqn paper: Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), 
the agent ensemble learn not much. The parameters can be found in maddpg/parameters in the object Par() and 
are assumed to be as follows:

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
        self.update_every = 16  # timesteps between updates
        self.num_updates = 1  # num of update passes when updating
        self.epsilon = 1.0  # epsilon for the noise process added to the actions
        self.epsilon_decay = 0  # decay for epsilon above
        self.num_episodes = 1000  # number of episodes
        
The result is not so good, as depicted in the illustration. The network learns almost nothing. 
![Alt text](results/paper_parameters.png?raw=true "Title")

The following changes were experimentally changed in the ParReacher object to reach a higher score: 


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
        self.file_name = 'Reacher_Linux_NoVis/Reacher.x86_64'

        self.update_every = 1  # timesteps between updates
        # self.num_updates = 16  # num of update passes when updating

        self.epsilon_decay = 1e-6  # decay for epsilon above

![Alt text](reacher-optimized-parameters.png?raw=true "Title")


While the above configuration is optimized for fast learning, the algorithm does not seem to converge without significant
variance, as seen in the individual scores and the standard deviation. 
To mitigate this, the following changes lead to a slower, but more thorough learning resulting in much less variance at
the end. The Ohrnstein-Uhlenbeck noise is not decayed and the policy gets stronger. 
These parameters taken from episode 200 are the considered the best parameters. 
Using watch_reacher.py, the quick learner indeed showed some errors in reaching the target. The slowly learned network
showed only two slower response times to reach the target but was flawlessly everywhere else.
This is seen in the variance of the score. 

        self.update_every = 4  
        self.epsilon_decay = 1e-6  # decay for epsilon which is a reduction factor for the ounoise


![Alt text](results/slow_learning_least_variance.png?raw=true "Title")


The network architecture is as follows: 

The actor uses the configuration similar to the ddpg paper, but with 128x128 weights in the fully connected layer. 
Because the physical dimensions of the input layer are quite different in range a batch normalization prevents numerical
instability. 

    Actor(
      (bn0): BatchNorm1d(33, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1): Linear(in_features=33, out_features=128, bias=True)
      (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc2): Linear(in_features=128, out_features=128, bias=True)
      (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc3): Linear(in_features=128, out_features=4, bias=True)
    )
The cricic also uses the reference network with a reduced number of neural network weights. 

    Critic(
      (bn0): BatchNorm1d(33, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fcs1): Linear(in_features=33, out_features=128, bias=True)
      (fc2): Linear(in_features=132, out_features=128, bias=True)
      (fc3): Linear(in_features=128, out_features=1, bias=True)
    )

# Future work

Results could be improved by using a policy-based black box learning algorithm like hillclimbing or 
[evolution strategies](https://openai.com/blog/evolution-strategies/) to improve the meta parameters. The parameter 
exploration helps to evaluate the maximum performance of the algorithm. 
As DDPG is similar to dqn, improvements like priority experience replay can further help improving the score. 
Furthermore, it would be intereseting to use further [ algorithms]()
 

# Background
## Policy-based methods 

### Definition of policy-based vs. value-based: 

Value-based methods use experienced state-action-reward-state tuples with the environment to estimate the optimal action-value 
function. The optimal policy is derived by choosing a policy which maximizes the expected value. 
Policy-based methods on the other hand directly learn the optimal policy, without maintaining the value estimate. 

Deep reinforcement learning represents the policy within a neural network. The input is the observed environment state. 
For a discrete output, the layer has a node for each possible action which shows the execution probability. 
At first, the network is initialized randomly. Then, the agent learns a policy as it interacts with the environment. 

Policy-based methods can learn either stochastic or deterministic policies, and the action space can either be finite 
or continous.

## Policy-based Methods for reinforced learning

### Hill Climbing

Hill climbing iteratively finds the weights θ for an optimal policy. 
For each iteration: 
- The values are slightly changed to yield a new set of weights
- A new episode is collected. If the new weights yield a higher reward, these weights are set
as the current best estimate. 

### Improving Hill Climbing 

- Steepest ascent hill climbing choses a small number of neighbouring policies at each iteration and chooses 
the best among them. This helps finding the best way towards the optimum. 
- Simulated annealing uses a pre-defined radius to control how the policy space is explored, which is reduced while closing in 
on the optimal solution. This makes search more efficient. 
- Adaptive noise scaling decreases the search radius with each improvement of a policy, while increasing the radius if no improvement was found. 
This makes it likely that a policy doesn't get stuck in a local optimum.    

### Methods beyond hill climbing: 
- The cross-entropy method iterates over neighbouring policies and uses the best performing policies to calculate a new estimate. 
- The evolution strategies method considers the return of each candidate policy. A reward-weighted sum over all candidate policies 
uses all available information. 

### Why policy-based methods? 
- Simplicity: Policy-based methods directly estimate the policy without the intermitting value function. This is more efficient 
in particular with respect to a vast action-space in which an estimate of each action has to be kept.  
- Stochastic policies: Policy-based methods learn true stochastic policies 
- Continous action spaces: Policy-based methods are well-suited for continous action spaces withouth the need for discretization. 

## Policy-gradient methods

In contrast to policy-based methods, policy-gradient methods use the gradient of the policy. They are a subclass of 
policy-based methods. The policy is nowadays often a neural network, in which the gradient is used to search for the 
optimal weights. 

The policy gradient method will use trajectories of state-action-reward-nextstate-nextaction to make actions with 
higher expected reward more likely.

A trajectory tau is a state-action sequence. The goal can be further clarified as the gradient is used to maximise the
expected reward of a given (set of) trajectories, because it is inefficient or impossible to compute the real gradient of all
possible trajectories. 

### Reinforce

The commonly used first policy-gradient method is reinforce. It uses the policy to collect a set of m trajectories with 
a horizon H. These are used to estimate the gradient and thus update the weights of the policy-neural network. This is 
repeated until a satisfactory score is achieved. 

This algorithm can solve MDPs with either stochastic or deterministic, discrete or continous action spaces. The latter was difficult with value-based
methods because the action needs to be discretized. DQNs can only use stochastic actions by leveraging the exploration factor, which is neither efficient nor pretty. 

### Proximal policy optimization (PPO)

Reinforce can be optimized using the following elements: Noise Reduction, Rewards Normalization, Credit Assignment, 
Importance Sampling with re-weighting to come up with the PPO algorithm. 
So that’s it! We can finally summarize the PPO algorithm

    - Collect some trajectories using the current policy π_θ
    - Initialize theta prime 
        θ′=θ
    - Compute the gradient of the clipped surrogate function using the trajectories
    - Update θ′ using gradient ascent 
        θ′←θ′+α∇θ′L_min_clipped_surrogate(θ′,θ)
    - Then we repeat step 2-3 without generating new trajectories. Typically, step 2-3 are only repeated a few times
    - Set theta θ=θ′, go back to step 1, repeat.


## Actor-critic methods

There are many different actor critic methods. Typically, the actor is a network computing an action, which the critic 
evaluates with assuming the state-value, state-action-value, or advantage of the gained state transition. 
The advantage is the difference in value of the states plus the gained reward. 
The actor policy-based network is then trained using the value the critic assumes. 
Therefore, the actor is a policy method and the critic is a value method. The critic helps to reduce the variance of 
policy-gradient method which have a high variance. 

The value-based critic can use a td-method or a monte-carlo method to estimate the value. A vanilla td-estimate means, 
that a single step, containing state, action, reward, next_state, next_action tuples are used to compute a gradient
to train the network. 
A monte-carlo estimate samples one trajectory of the game, then uses the reward to train the value-based network. 
Monte Carlo methods wait until the real reward of a trajectory is known, therefore they are unbiased. Since a trajectory
consists of many steps which in sum make up the rewards, a lot of variance is in the estimate. Imagine using entire
chess games to evaluate the value of one step. There are so many influences, that the variance is high. 
A TD-method has less variance, but is biased because an estimate is used to learn the estimate. 
A good method combines the least amount of variance with that amount of bias that is still capable of learning the
 optimal function. 
One compromise is an n-step td-method called n-step bootstrapping, in which n steps are used to estimate the value. Typically 5-6 steps are good,
but it varies across the problems to be solved. 
[Generalized advantage estimation (GAE)](https://arxiv.org/abs/1506.02438) can use a parameter to interpolate between td-estimate and monte-carlo-estimate
using a weighing factor. Depending on this weight, the future steps will be decayed, therefore a gradient is built 
using a mixture of the two.

Examples for actor-critic networks are: 
- A3C: Asynchronous Advantage Actor-Critic, N-step Bootstrapping
- A3C: Asynchronous Advantage Actor-Critic, Parallel Training
- A2C: Advantage Actor-Critic
- DDPG: Deep Deterministic Policy Gradient, Continuous Action-space

## DDPG

DDPG is an untypical actor-critic method and it could be seen as approximate dqn. The reason for this is that the critic
 in the DDPG is used to approximate the maximizer over the q-values of the next state and not as a learned baseline as
  in typical actor-critic methods. But this is still a very important algorithm. 
  
A limitation of the dqn agent is that it is not straightforward to use in continous deterministic action space. Discretizing
a continous action-space suffers from the curse of dimensionality and does not scale well. 
In DDPG two neural networks are used. One is the actor policy which computes an action based on both the state and the 
neural weights. The critic computes a state-value based on the neural weights. 
 The actor is learning the argmax Q(s,a) which is the best action. Therefore it is a deterministic policy. 
 The critic learns to evaluate the optimal action-value function by using the actors best believed action. 
 To help optimization, noise to the action-vector is used to help exploration. 
 
 
 