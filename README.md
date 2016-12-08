# OpenAI Gym ZOO

This repository is home for some artificial intelligence agents and other creatures that tries to solve [OpenAI Gym Challanges](https://gym.openai.com/)



### [CartPole_AI_1.py](https://github.com/maitek/openai-gym-zoo/blob/master/CartPole_AI_1.py) 

This AI agent tries to solve the [CartPole Challenge](https://gym.openai.com/envs/CartPole-v0) by using a feed forward neural network, trained using a stochastic optimization algorithm know as [Cross-Entropy Method](https://en.wikipedia.org/wiki/Cross-entropy_method). 

The training starts by randomly instantiate population of agents. Each agent consist of a feed-forward neural network with one hidden layer. Each AI tries balance the stick as long as possible, by evaluating the observed scene and deciding to move left or right at each frame. The observed parameters are forward-feed through the neural network which outputs probabilities of moving left or right. The decision is then chosen randomly weighted by the probabilities.

Each agents plays a number of games and are then ranked by their average score. The agents that scored the best (i.e. could balance the longest without falling) are chosen as parents for the next generation. The weights of the neural network for the next generation are sampled from a Gaussian distribution around the weights of the parents. This way each generation the neural network weights are tweaked to perform better than the previous generation.

This agent usually manages to solve the challange after ca 1000-2000 episodes.

### [CartPole_AI_2.py](https://github.com/maitek/openai-gym-zoo/blob/master/CartPole_AI_2.py)

This agent uses a more sophisticated learning algorithm using policy gradients. This agent implements a variation of the commonly known REINFORCE algorithm in order to solve the [CartPole Challenge](https://gym.openai.com/envs/CartPole-v0). In contrast with the CEM method that only learns using the outcome of the full game this agent evaluates every frame if the decision i.e. moving left or right given the observations, was good or bad. By estimating the expected reward for each frame the agent can optimize its policy to maximaize the total reward. 

The policy esitmator is a neural netwokr that is trained using stochastic gradient decent, using the ADAM optimzer. The decision is stochastically sampled from the policy distribution output by the neural network.

#### Modification to REINFORCE: 

The neural network is updated after each played episode. To maximize the utilization the known information, the neural network is also trained again using previous episode, but the rewards from every previous episode is decayed exponentially towards zero. This way agent can remember decisions that it made a few episodes back, and use that for training. This modification to the standard REINFORCE algorithm seemes make the agent learn 1.5-2x faster.

This agent usually manages to solve the challange after ca 100-200 episodes.


## Additional Material
OpenAI Gym Reinforcement Learning Tutorial
https://gym.openai.com/docs/rl


