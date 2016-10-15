# OpenAI Gym ZOO

This repository is home for some artificial intelligence algorithms and other creatures that tries to solve [OpenAI Gym Challanges](https://gym.openai.com/)


### CartPole_AI_1

This AI algorithm tries to solve the [CartPole Challenge](https://gym.openai.com/envs/CartPole-v0) by using a feed forward neural network, trained using a stochastic optimization algorithm know as [Cross-Entropy Method](https://en.wikipedia.org/wiki/Cross-entropy_method). This is a genetic algorithm inspired by evolution.

The training starts by randomly instantiate population of AI. Each AI consist of a feed-forward Neural network with one hidden layer. Each AI tries solve balance the stick as long as possible, by evaluating the observed scene and deciding to move left or right at each frame. The observed parameters are forward-feed through the neural network which outputs probabilities of moving left or right. The decision is then chosen randomly weighted by the probabilities.

Each AI plays a number of games and are then ranked by their average score. The AIs that scored the best (i.e. could balance the longest without falling) are chosen as parents for the next generation. The weights of the neural network for the next generation are sampled from a Gaussian distribution around the weights of the parents. This way each generation the neural network weights are tweaked to perform better than the previous generation.





