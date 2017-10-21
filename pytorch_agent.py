import gym
from itertools import count
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class DiscreteAgent():
    def __init__(self,input_size, possible_actions):
        self.possible_actions = possible_actions
        self.policy = PolicyNet(input_size, output_size=len(possible_actions))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

    def decide_action(self, observation):
        # predict probabilities for actions
        var_s = Variable(torch.from_numpy(observation.astype(np.float32)))
        action_prob = torch.exp(self.policy.forward(var_s))
        # select random action weighted by probabilities
        action =  np.random.choice(self.possible_actions, 1, p=action_prob.data.numpy())[0]
        return action

    def discount_rewards(self,r,gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            running_add = running_add * gamma + r[t][0]
            discounted_r[t][0] = running_add
        return discounted_r

    def iterate_minibatches(self, observation, actions, rewards,  batchsize, shuffle=False):
        assert len(observation) == len(actions)
        assert len(observation) == len(rewards)

        indices = np.arange(len(observation))
        if shuffle:
            np.random.shuffle(indices)
        #import pdb; pdb.set_trace()
        for start_idx in range(0, len(observation), batchsize):
            if shuffle:
                excerpt = indices[start_idx:min(start_idx + batchsize, len(indices))]
            yield observation[excerpt], actions[excerpt], rewards[excerpt]

    def update(self,observation, actions, rewards):
        # discounted reward
        rewards = self.discount_rewards(summary["rewards"], gamma = 0.99)
        self.optimizer.zero_grad()
        # L = log π(a | s ; θ)*A
        loss = 0
        for observation_batch, action_batch, reward_batch in self.iterate_minibatches(observation, actions, rewards, batchsize = 100, shuffle=True):
            #import pdb; pdb.set_trace()
            s_var =  Variable(torch.from_numpy(observation_batch.astype(np.float32)))
            a_var = Variable(torch.from_numpy(action_batch).view(-1))
            A_var = Variable(torch.from_numpy(reward_batch.astype(np.float32)))
            pred = self.policy.forward(s_var)
            loss += F.nll_loss(pred+torch.log(A_var),a_var)

        loss.backward(loss)
        self.optimizer.step()





class CartPole():
    def __init__(self,render=True, max_frames = 1000):
        self.env = gym.make('CartPole-v0')
        self.render = render
        self.max_frames = max_frames
        self.episode = dict()

    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return self.env.observation_space.shape[0]

    def num_actions(self):
        return self.env.action_space.n

    def play(self,agent):
        """ Play one episode and collect observations and rewards """

        summary = dict()
        summary['rewards'] = list()
        summary['observations'] = list()
        summary['actions'] = list()
        observation = self.env.reset()
        t = 0

        for t in range(self.max_frames):
            if(self.render):
                self.env.render()

            action = agent.decide_action(observation)

            summary['observations'].append(observation)
            summary['actions'].append(action)
            observation, reward, done, info = self.env.step(action)
            summary['rewards'].append(reward)

            if done:
                break

        summary['observations'] = np.vstack(summary['observations'])
        summary['actions'] = np.vstack(summary['actions'])
        summary['rewards'] = np.vstack(summary['rewards'])
        return summary




if __name__ == "__main__":

    env = CartPole(render=True,max_frames = 1000)
    agent = DiscreteAgent(
        input_size=env.num_observations(),
        possible_actions=env.get_actions()
    )

    for episode_idx in range(1,10000):

        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])
        print("Episode {}, mean score {}".format(episode_idx,summary['score']))

        # Update agent
        agent.update(summary["observations"],summary["actions"],summary["rewards"])
