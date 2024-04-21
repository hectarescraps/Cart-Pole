# Cart-Pole
A cart pole balancing agent powered by Q-Learning. Utilizes Python 3 and Gymnasium (formerly OpenAI Gym).

My first attempt at a solution leveraged a static learning rate which did not result in strong performance - 20,000 Episodes: 140 Mean Score. This can be seen in the file [cartpole_static.py.](https://github.com/hectarescraps/Cart-Pole/blob/main/cartpole_static.py)

To increase performance, I transitioned to a variable learning rate which decays as the number of training episodes increases. This led to drastically improved results – 20,000 Episodes: 1,246 Mean Score ([code](https://github.com/hectarescraps/Cart-Pole/blob/main/cartpole.py), [trained Q Table](https://github.com/hectarescraps/Cart-Pole/blob/main/qtable.pickle)). Note: a mean score over 200 is considered a successful solution.  

Please feel free to play with / adjust my Q Learning implementation. If you yield a better result, please let me know - I'd love to understand the changes you made and why. Thank you so much for reading!

A few notes on Q Learning and my implementation - 

## _High Level Overview of Q Learning_

Given a state in an environemnt, Q Learning explores different actions which it can take in this state, observes the reward associated with these actions, and updates its knowledge of the perceived quality of these state-action pairs. After sufficient exploration, Q Learning produces a trained model to choose the 'optimal' action in any given state, leading to the greatest possible reward. Optimally is in quotes here as this heavily depends on the specific implementation of Q Learning. 

Q Learning accomplishes this with no prior assumptions or knowledge of the environment. Hence, Q Learning is referred to as a model-free Reinforcement Learning algorithm.

In the Cart Pole scenario, a given state is represented by: the position of the cart, the cart's velocity, the angle of the pole, and the pole's velocity. The potential actions are simply: move the cart left or move the cart right.

For more information on Q Learning, please see: https://en.wikipedia.org/wiki/Q-learning

## _Epsilon Greedy_

In Q Learning, we want to first explore the environment, and then apply our observed understanding of the environment to arrive at an optimal policy (that is, the action we should take in any given state to maximize our reward). To do this, we must balance our competing desires to explore and exploit (i.e., maximize rewards in) the environment. Epsilon Greedy is a framework to strike such a balance and efficienctly converge to an optimal policy. 

## _Binning_

Q Learning represents a given environment as a finite number of state-action pairs, but the Cart Pole environment has a continuous state space (i.e., an infinite number of states). Thus, to apply Q Learning to the Cart Pole environment, we must first discretize or bin the Cart Pole state space. This is accomplished with the cart_pos_space, cart_velo_space, pole_ang_space and pole_velo_space parameters (at the top of [cartpole.py](https://github.com/hectarescraps/Cart-Pole/blob/main/cartpole.py)) and the Bin function (seen in [helper.py](https://github.com/hectarescraps/Cart-Pole/blob/main/helper.py))



