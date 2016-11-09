import gym
env = gym.make('CartPole-v0')
from itertools import count
import numpy as np
import json

POPULATION_SIZE = 100
SURVIVE_RATIO = 0.1
HIDDEN_UNITS = 20
NUM_GAMES_PER_AI = 1
SIGMA_FACTOR = 10
RENDER = True
WEIGHT_POPULATION = True

# Simple policy network with 1 hidden layer
class PolicyNetwork(object):
    _ids = count(0)
    
    def __init__(self):
        self.id = next(self._ids)
        self.D = 4 # Input variables
        self.H = HIDDEN_UNITS # number of hidden layer neurons
        self.network = dict()
        self.network['W1'] = np.random.randn(self.H,self.D) / np.sqrt(self.D) # "Xavier" initialization   
        self.network['W2'] = np.random.randn(self.H) / np.sqrt(self.H)
        self.score = 0

    def update_from_population(self, population, population_weights = None, sigma_factor = SIGMA_FACTOR):
        if population_weights is None:
            population_weights = np.ones((len(population)))

        W1 = np.zeros((len(population),self.H,self.D))
        W2 = np.zeros((len(population),self.H))
        for idx, p in enumerate(population):
            W1[idx,:,:] = p.network['W1'] 
            W2[idx,:] = p.network['W2']

        W1_mu = np.average(W1, weights = population_weights, axis=0) 
        W1_sigma = np.std(W1,axis=0) * sigma_factor
        self.network['W1'] = W1_sigma*np.random.randn(self.H,self.D) + W1_mu

        W2_mu = np.average(W2, weights = population_weights, axis=0)
        W2_sigma = np.std(W2,axis=0) * sigma_factor
        self.network['W2'] = W2_sigma*np.random.randn(self.H) + W2_mu


    def policy_forward(self,x):
        h = np.dot(self.network['W1'], x)
        # ReLU nonlinearity
        h[h<0] = 0 
        logp = np.dot(self.network['W2'], h)
        # sigmoid
        p = 1.0 / (1.0 + np.exp(-logp))
        return p, h # return probability of taking action 2, and hidden state

    def decide_action(self,observation):
        p,h  = self.policy_forward(observation)
        action = 0 if np.random.uniform() < p else 1 
        return action


# Play an episode using model
def play_episiode(model, max_frames = 1000, render = False, full = False):
    observation = env.reset()
    t = 0
    for t in range(max_frames):
        if(render):
            env.render()
           
        action = model.decide_action(observation)

        observation, reward, done, info = env.step(action)
        
        if done and not full:
            #print("Model {} finished after {} timesteps".format(model.id,t+1))
            break
    return t

print(env.observation_space)
seed_population = [PolicyNetwork() for x in range(0,POPULATION_SIZE)]
population_weights = None



for generation in range(1,10000):
    print("Generation {}".format(generation))
    # Generate new population
    population = [PolicyNetwork() for x in range(0,POPULATION_SIZE)]
    for p in population:
        p.update_from_population(seed_population, population_weights = population_weights) 

    for model in population:
        # each AI plays a number of games, and average score counts
        scores = []
        for i in range(0,NUM_GAMES_PER_AI):
            score = play_episiode(model, render = False)
            scores.append(score)
        model.score = np.mean(scores)
        
    # find population with best score
    print("Population mean score", np.mean([x.score for x in population]))
    
    
    survive_size = int(SURVIVE_RATIO*len(population))
    population = sorted(population, key=lambda k: k.score,reverse=True)[0:survive_size]
    print("Best players",[x.score for x in population])

    if WEIGHT_POPULATION:
        population_weights = [x.score for x in population]
        population_weights = population_weights/np.sum(population_weights)
        print("Weights",population_weights)
    



    # Test if the best player can beat the challenge 100 games with score higher than 195
    test_failed = False
    score = play_episiode(population[0], render = RENDER, max_frames = 1000)
    for i in range(0,100):
        score = play_episiode(population[0], render = False)
        print("Test game {}, score {}".format(i,score))
        if score < 195.0:
            test_failed = True
            break

    if not test_failed:
        print("Test passed!")
        break
    else:
        print("Test failed!")

    seed_population = population

