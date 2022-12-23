import gym
import numpy as np
from tqdm import tqdm

RENDER = False

env = gym.make('CartPole-v1')
# env.reset()  # Environment reset
# if RENDER: env.render()  # Environment render to pygame window

gamma = 0.9

w = np.random.rand(2, 5)
w -= 0.5

max_episode = 10000
max_step = 500
alpha = 0.01

for epi in tqdm(range(max_episode)):
    env.reset()
    if RENDER: env.render()
    for step in range(max_step):
        before_state = np.array(env.state)
        
        action = np.zeros(2)
        # gibbs softmax function
        for act in range(2):
            action[act] = np.dot(w[act, 1:], before_state) + w[act, 0]
        pr = np.zeros(2)
        for i in range(2):
            pr[i] = np.exp(action[i]) / np.sum(np.exp(action))
        action = np.random.choice(2, 1, p=pr)[0]

        observation, reward, done, _ = env.step(action)
        if RENDER: 
            env.render()
        

        next_act = np.zeros(2)
        for act in range(2):
            next_act[act] = np.dot(w[act, 1:], observation) + w[act, 0]
        best_action = np.argmax(next_act)

        now_q = np.dot(w[action, 1:], before_state) + w[action, 0]
        next_q = np.dot(w[best_action, 1:], observation) + w[best_action, 0]
        
        w[action, 0] += alpha * (reward + gamma * next_q - now_q)
        w[action, 1:] += alpha * (reward + gamma * next_q - now_q) * before_state

        if done == True:
            break
    if epi == max_episode - 100:
        RENDER = True