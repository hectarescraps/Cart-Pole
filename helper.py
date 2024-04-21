import random
import numpy as np

""" Helper Functions """

# helper function to transform continuous env space outputs to discretized buckets

def Bin(state, observation_space_1, observation_space_2, observation_space_3, observation_space_4):
    cart_pos_bin = np.digitize(state[0], observation_space_1)
    cart_velo_bin = np.digitize(state[1], observation_space_2)
    pole_ang_bin = np.digitize(state[2], observation_space_3)
    pole_velo_bin = np.digitize(state[3], observation_space_4)
    binned_state = (cart_pos_bin, cart_velo_bin, pole_ang_bin, pole_velo_bin)
    return binned_state

# helper function to return initialized q_table of 0's
## increment observation space variables by 1 to account for positive outlier observations (e.g., cart position > 2.4)

def InitQTable(env, observation_space_1, observation_space_2, observation_space_3, observation_space_4):
     q_table = np.zeros([len(observation_space_1) + 1, len(observation_space_2) + 1, len(observation_space_3) + 1, len
            (observation_space_4) + 1, env.action_space.n])
     return q_table

# update q table based on most recent action

def UpdateQ(q_table, learning_rate, discount_rate, old_state, new_state, action, reward):
    q_table[old_state][action] = q_table[old_state][action] + learning_rate*(reward + discount_rate*np.max(q_table[new_state]) - q_table[old_state][action])

# epsilon greedy to handle explore/exploit tradeoffs. 
## function returns corresponding action

def EpsilonGreedy(epsilon, env, binned_state, q_table): 
    p = random.uniform(0, 1)
    if p > epsilon: 
        action = np.argmax(q_table[binned_state])
    else: 
        action = env.action_space.sample()
    return action

# greedy function to utilize post-training. 
## function returns corresponding action

def Greedy(binned_state, q_table):
    action = np.argmax(q_table[binned_state])
    return action 
