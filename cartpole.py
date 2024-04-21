import gymnasium as gym
import numpy as np
from tqdm import trange
import helper as help
import pickle

""" defining file controls and initializing the environment """

# visual_testing: 
## True: displays rendered episodes of the trained model for user to analyze
## False: skips this step
visual_testing = True

# initializing the environment
env = gym.make("CartPole-v1", max_episode_steps=2000)

""" discretizing the observation space """

# cart position range 
cart_pos_space = np.linspace(-2.4, 2.4, 8)

# in practice cart velocity can range from (-4, 4)
cart_velo_space = np.linspace(-4, 4, 1)

# pole angle range 
pole_ang_space = np.linspace(-.2095, .2095, 10)

# pole velocity can range from (-4, 4)
pole_velo_space = np.linspace(-4, 4, 12)

""" model parameters """

training_episodes = 20000
evaluation_episodes = 1000
visual_testing_episodes = 3

epsilon_decay = -.0001
max_epsilon = 1
min_epsilon = 0.1

learning_rate_decay = -.001
max_learning_rate = 1
min_learning_rate = 0.1

discount_rate =  0.99

training_seeds = list(map(int, np.random.randint(0, 1000000, training_episodes)))
evaluation_seeds = list(map(int, np.random.randint(0, 1000000, evaluation_episodes)))
visual_testing_seeds = list(map(int, np.random.randint(0, 1000000, visual_testing_episodes)))

""" Core Model Implementation Function handling Training, Visualizing and Evaluating
         Returns: (q_table, [episode scores]) """

# **kwargs apply to model training (train=True)
### **kwargs to define: max_epsilon, min_epsilon, epsilon_decay, max_learning_rate, min_learning_rate, learning_rate_decay, discount_rate
def Play(q_table, episodes, seeds, env=None, train=False, visual=False, print_progress=True, **kwargs):
    track = []
    # if visual, create new rendered environment for analysis. otherwise proceed.
    if visual: env = gym.make("CartPole-v1", render_mode = 'human', max_episode_steps=2000)
    
    # loop over the number of designated training epsiodes
    for episode in trange(episodes):
        
        # if training, reduce epsilon, learning rate. these reductions do not apply to train=False cases
        if train:
            epsilon = kwargs['min_epsilon'] + (kwargs['max_epsilon'] - kwargs['min_epsilon'])*np.exp(kwargs['epsilon_decay']*episode) 
            learning_rate = kwargs['min_learning_rate'] + (kwargs['max_learning_rate'] - kwargs['min_learning_rate'])*np.exp(kwargs['learning_rate_decay']*episode) 
        
        # printing mean of prior 100 episodes every 1,000 episodes to track model progress
        if print_progress and episode != 0 and episode % 1000 == 0:
            print('Episode ' + str(episode) + '- Prior 100 Mean: ' + str(np.mean(track[episode - 100:])))
        
        # if training, printing epsilon, learning rate every 1,000 episodes for reference
        if train and episode != 0 and episode % 1000 == 0:
            print('Episode ' + str(episode) + '- Learning Rate: ' + str(learning_rate) + ' Epsilon: ' + str(epsilon))
    
        # in all cases, reset environment, init cumulative reward tracker    
        state, _ = env.reset(seed=seeds[episode])
        binned_state = help.Bin(state, cart_pos_space, cart_velo_space, pole_ang_space, pole_velo_space)
        terminated, truncated = False, False
        cumulative_reward = 0
        
        # loop while env is not terminated or truncated
        while not(terminated or truncated):
            
            # determine action via epsilon greedy / greedy function, take action
            if train: action = help.EpsilonGreedy(epsilon, env, binned_state , q_table)
            else: action = help.Greedy(binned_state, q_table)
            new_state, reward, terminated, truncated, _ = env.step(action)
            binned_new_state = help.Bin(new_state, cart_pos_space, cart_velo_space, pole_ang_space, pole_velo_space)
            
            # if training, update q_table
            if train: help.UpdateQ(q_table, learning_rate, kwargs['discount_rate'], binned_state, binned_new_state, action, reward)
            
            # update cumulative reward, set binned_state = binned_new_state
            cumulative_reward += 1
            binned_state = binned_new_state
        
        track.append(cumulative_reward)    
    return (q_table, track)

""" defining main function """

def main(): 
    # initialize q_table 
    q_table = help.InitQTable(env, cart_pos_space, cart_velo_space, pole_ang_space, pole_velo_space)

    # train model
    trained_table  = Play(q_table, training_episodes, training_seeds, env=env, train=True, max_epsilon=max_epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, max_learning_rate=max_learning_rate, min_learning_rate=min_learning_rate, learning_rate_decay=learning_rate_decay, discount_rate=discount_rate)[0]

    # visual analysis of model if visual_test = True
    if visual_testing: 
        Play(trained_table, visual_testing_episodes, visual_testing_seeds, visual=True, print_progress=False)

    # evaluate greedy policy, return mean of 1000 scores
    scores = Play(trained_table, evaluation_episodes, evaluation_seeds, env=env, train=False)[1]
    print(np.mean(scores))

    # save q_table for future reference
    with open('qtable.pickle', 'wb') as f:
        pickle.dump(trained_table, f)
    
if __name__ == '__main__':
    main()









        
        



            




    



        
        
        
        
        



    



