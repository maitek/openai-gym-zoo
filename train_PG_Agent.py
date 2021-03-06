import gym
import numpy as np
from agents.utils import play_full_episiode
from agents.agents import PolicyNetwork



def discount_rewards(r,gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.shape[0])):
        running_add = running_add * gamma + r[t][0]
        discounted_r[t][0] = running_add
    return discounted_r



# Create environment
env = gym.make('CartPole-v0')
possile_actions = range(env.action_space.n)
size_observations = sum(env.observation_space.shape)
print(possile_actions,size_observations)


observations_all = None
rewards_all = None
actions_all = None
scores_all = None

# simulation parameters
max_frames = 5000
max_frames_test = 200

# hyper parameters
learning_rate = 0.1
discount_factor = 0.99
episode_discount = 0.95
replay_buffer_len = 1000000

# book keeping
total_episodes = 0
total_trajectories = 0


# Initilize agent
agent = PolicyNetwork(size_observations=size_observations, possible_actions=possile_actions, learning_rate=learning_rate)

for episode in range(1,1000):

    # play episodes
    episodes = list()

    for i in range(0,1):

        episode = play_full_episiode(env, agent, max_frames = max_frames, render = True)

        # discounted reward
        episode["discount_rewards"] = discount_rewards(episode["rewards"], gamma = discount_factor)

        episode['score'] = np.sum(episode["rewards"])
        episodes.append(episode)

        total_episodes += 1
        total_trajectories += len(episode['rewards'])

    # concatenate episodes
    observations_new = np.vstack([x["observations"] for x in episodes])
    rewards_new = np.vstack([x["discount_rewards"] for x in episodes])
    actions_new = np.vstack([x["actions"] for x in episodes])
    scores_new = np.vstack([x["score"] for x in episodes])
    #print(observations_new)


    episode_bonus = np.sum(scores_new)

    observations_all = observations_new if observations_all is None else np.vstack((observations_all,observations_new))[:replay_buffer_len,:]
    rewards_all = rewards_new if rewards_all is None else np.vstack((rewards_all*episode_discount,rewards_new*episode_bonus))[:replay_buffer_len,:]
    actions_all = actions_new if actions_all is None else np.vstack((actions_all,actions_new))[:replay_buffer_len,:]

    scores_all = scores_new if scores_all is None else np.vstack((scores_all,scores_new))

    #print(rewards_all.shape)
    #rewards_all = rewards_all[rewards_all < reward_eps][:]
    #print(rewards_all.shape)

    print("Episode {}, Trajectories {}, mean score {}".format(total_episodes,
        total_trajectories, np.mean(scores_new)))

    # update weights
    agent.update_network(observations_all,actions_all,rewards_all)


    # === TEST ===
    # Test if the best player can beat the challenge 100 games with score higher than 195
    pass_score = 195.0
    pass_episodes = 100

    if episode['score'] > pass_score:
        test_failed = False
        for i in range(0,100):
            episode = play_full_episiode(env, agent, max_frames=max_frames_test, render = False)
            score = np.sum(episode["rewards"])
            print("Test game {}, score {}".format(i,score))
            if score < pass_score:
                test_failed = True
                break
        if not test_failed:
            print("Test passed!")
            break
        else:
            print("Test failed!")
