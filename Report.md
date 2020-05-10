[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Reaching bulbs with MADDPG

## Report



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
 
