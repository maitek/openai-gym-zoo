import numpy as np

"""
    Plays a full episode of OpenAI Gym environment and return an
    episode dictionary with the collected observations, actions and rewards
"""
def play_full_episiode(env, agent, max_frames = 1000, render = False):

    episode = dict()
    episode['rewards'] = list()
    episode['observations'] = list()
    episode['actions'] = list()

    observation = env.reset()
    t = 0
    for t in range(max_frames):
        if(render):
            env.render()

        action = agent.decide_action(observation)
        episode['observations'].append(observation)
        episode['actions'].append(action)
        observation, reward, done, info = env.step(action)
        episode['rewards'].append(reward)

        if done:
            break

    #print("Episode finished after {} timesteps".format(t+1))
    # convert to numpy array
    episode['observations'] = np.vstack(episode['observations'])
    episode['actions'] = np.vstack(episode['actions'])
    episode['rewards'] = np.vstack(episode['rewards'])
    return episode
