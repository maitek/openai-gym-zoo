from itertools import count
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import softmax, rectify

class Agent(object):

    def __init__(size_observations, possible_actions):
        self._possible_actions = possible_actions
        self._size_observations = size_observations

    def decide_action(self,observation):
        return np.random.choice(possible_actions)

class PolicyNetwork(Agent):
    def __init__(self, size_observations, possible_actions, learning_rate = 0.5):
        # List all possible action
        self._possible_actions = possible_actions
        self._size_observations = size_observations
        self._learning_rate = learning_rate

        # Prepare Theano variables for inputs and targets
        observation_var = T.matrix('observations')
        action_var = T.ivector('actions')
        reward_var = T.fvector('reward')
        learning_rate = T.scalar('learning_rate')

        # Build network structure
        self._network = self._build_network(observation_var,size_observations,len(possible_actions))
        print(self._network)

        probs = lasagne.layers.get_output(self._network)

        # Define Policy Loss function that increases propability of good action
        log_policy = lasagne.objectives.categorical_crossentropy(probs, action_var)
        loss = T.dot(log_policy,reward_var)

        # Calculate gradient
        params = lasagne.layers.get_all_params(self._network, trainable=True)
        grads = theano.grad(loss,params)

        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)

        self._update_network = theano.function([observation_var, action_var, reward_var, learning_rate], loss, updates=updates, on_unused_input='warn')

        # Define Theano forward function pass for prediction
        test_prediction = lasagne.layers.get_output(self._network, deterministic=True)
        self._forward_pass = theano.function([observation_var], test_prediction)


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

    # Update network parameters
    def update_network(self,observations,actions,rewards):

        S = observations.astype(np.float32)
        a = actions.astype(np.int32)
        r = rewards.astype(np.float32)

        # normalize awards
        r /= np.std(r)
        r -= np.mean(r)
        r[r < 0] = 0

        train_loss = 0
        for batch in self.iterate_minibatches(S,a,r, batchsize = 100, shuffle=True):
            _S, _a, _r = batch
            loss = self._update_network(_S, np.squeeze(_a), np.squeeze(_r), self._learning_rate)
            train_loss += loss

        train_loss /= len(r)
        #print(r)

        # decay learning rate
        #self._learning_rate = self._learning_rate*0.99
        #print(self._learning_rate)

        print("Training loss", train_loss)
        print("")

    # Choose an action depending on probabilities given an observation
    def decide_action(self,observation):
        X = np.vstack(observation).T
        probabilities  = self._forward_pass(X)[0]
        #print(probabilities)
        action = np.random.choice(self._possible_actions, 1, p=probabilities)[0]
        return action

    def iterate_minibatches(self, observation, actions, rewards,  batchsize, shuffle=True):
        assert len(observation) == len(actions)
        assert len(observation) == len(rewards)
        if shuffle:
            indices = np.arange(len(observation))
            np.random.shuffle(indices)
        for start_idx in range(0, len(observation) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield observation[excerpt], actions[excerpt], rewards[excerpt]
