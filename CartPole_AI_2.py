import gym
env = gym.make('CartPole-v0')
from itertools import count
import numpy as np
import json
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import softmax, rectify

class PolicyNetwork(object):
    def __init__(self, size_observations, possible_actions):
        # List all possible action
        self._possible_actions = possible_actions
        self._size_observations = size_observations

        # Prepare Theano variables for inputs and targets
        input_var = T.matrix('inputs')
        target_var = T.ivector('targets')
        reward_var = T.scalar('reward')
        

        # Build network structure
        self._network = self._build_network(input_var,size_observations,len(possible_actions))
        print(self._network)
        # Define Theano update function for training
        prediction = lasagne.layers.get_output(self._network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        #grad = # TODO define gradients manually
        

        params = lasagne.layers.get_all_params(self._network, trainable=True)
        grads = theano.grad(loss,params)
        

        grads_scaled = list()
        for g in grads:
            grads_scaled.append(g*reward_var)
        
        updates = lasagne.updates.nesterov_momentum(
                grads, params, learning_rate=0.001, momentum=0.9)

        self._update_network = theano.function([input_var, target_var, reward_var], loss, updates=updates,on_unused_input='warn')

        # Define Theano forward function pass for prediction
        test_prediction = lasagne.layers.get_output(self._network, deterministic=True)        
        self._forward_pass = theano.function([input_var], test_prediction)


    # Define network architecture
    def _build_network(self,input_var, num_inputs, num_outputs):
            input_shape = (None,num_inputs)
            #print(input_shape)
            input_layer = InputLayer(shape=input_shape, input_var=input_var)
            hidden_layer = DenseLayer(input_layer, num_units=40, 
                nonlinearity=rectify, W=lasagne.init.GlorotUniform())
            network = lasagne.layers.DenseLayer(hidden_layer,num_units=num_outputs,
                nonlinearity=softmax)
            return network

    def _discounted_rewards(self,r):
        dr = np.cumsum(r) #[::-1]

        return dr

    # Update network parameters 
    def update_network(self,observations,actions,rewards):
        
        X = observations.astype(np.float32)
        y = actions.astype(np.int32)
        r = rewards.astype(np.float32)

        # normalize awards
        r -= np.mean(r)
        r /= np.std(r)
        #print(r[:100])
        #exit()
        
        train_loss = 0
        for i in range(0,len(r)):
            _X = X[i:i+1,:]
            _y = y[i,:]
            _r = r[i][0]
            train_loss += self._update_network(_X, _y, _r)
        train_loss /= len(r)
        print("Training loss", train_loss)

    # Choose an action depending on probabilities given an observation
    def decide_action(self,observation):
        X = np.vstack(observation).T
        probabilities  = self._forward_pass(X)[0]
        #print(probabilities)
        action = np.random.choice(self._possible_actions, 1, p=probabilities)[0]
        return action

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]






#def discounted_rewards(r):
#    r = np.multiply(r,np.sum(r))
#    return r 

def discount_rewards(r,gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.shape[0])):
        running_add = running_add * gamma + r[t][0]
        discounted_r[t][0] = running_add
    return discounted_r


def play_episiode(model, max_frames = 1000, render = False):
    
    episode = dict()
    episode['rewards'] = list()
    episode['observations'] = list()
    episode['actions'] = list()

    observation = env.reset()
    t = 0
    for t in range(max_frames):
        if(render):
            env.render()
           
        action = model.decide_action(observation)
        episode['observations'].append(observation)
        episode['actions'].append(action)
        observation, reward, done, info = env.step(action)
        episode['rewards'].append(reward)

        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            # convert to numpy array
            episode['observations'] = np.vstack(episode['observations'])
            episode['actions'] = np.vstack(episode['actions'])
            episode['rewards'] = np.vstack(episode['rewards'])

            break
    return episode



model = PolicyNetwork(size_observations=4, possible_actions=[0,1])

for episode in range(1,1000):

    # play episodes
    episodes = list()

    for i in range(0,100):
        episode = play_episiode(model, render = False)
        # discounted reward
        episode["discount_rewards"] = discount_rewards(episode["rewards"], gamma = 0.9)
        #print("rew",episode["rewards"])
        #print("drew",episode["discount_rewards"])
        #exit()
        episode['score'] = np.sum(episode["rewards"])
        episodes.append(episode)
        
    # concatenate episodes
    observations_all = np.vstack([x["observations"] for x in episodes])
    rewards_all = np.vstack([x["discount_rewards"] for x in episodes])
    actions_all = np.vstack([x["actions"] for x in episodes])
    scores_all = np.vstack([x["score"] for x in episodes]) 

    print("Mean score", np.mean(scores_all))
    # update weights
    model.update_network(observations_all,actions_all,rewards_all)








"""        
    # find population with best score
    print("Population mean score", np.mean([x.score for x in population]))
    survive_size = int(SURVIVE_RATIO/len(population))
    population = sorted(population, key=lambda k: k.score,reverse=True)[0:10]
    print("Best players",[x.score for x in population])


    # Test if the best player can beat the challenge 100 games with score higher than 195
    test_failed = False
    score = play_episiode(population[0], render = True, max_frames = 200)
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
"""
