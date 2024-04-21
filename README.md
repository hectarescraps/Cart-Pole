# Cart-Pole
A cart pole balancing agent powered by Q-Learning. Utilizes Python 3 and Gymnasium (formerly OpenAI Gym).

My first attempt at a solution leveraged a static learning rate which did not result in strong performance - 20,000 Episodes: 140 Mean Score. This can be seen in the file cartpole_static.py.

To increase performance, I transitioned to a variable learning rate which decays as the number of training episodes increases. This led to drastically improved results – 20,000 Episodes: 764 Mean Score. Note: over 200 is considered a successful solution.

A few notes on Q Learning and my implementation - 

** High Level Overview of Q Learning **

Given a state in an environemnt, Q Learning explores different actions which it can take in this state, observes the reward associated with these actions, and updates its knowledge of the perceived quality of these state-action pairs. After sufficient exploration, Q Learning produces a trained model to choose the 'optimal' action in any given state, leading to the greatest possible reward. Optimally is in quotes here as this heavily depends on the specific implementation of Q Learning. 

Q Learning accomplishes this with no prior assumptions or knowledge of the environment. Hence, Q Learning is referred to as a model-free Reinforcement Learning algorithm.

In the Cart Pole scenario, a given state is represented by: the position of the cart, the cart's velocity, the angle of the pole, and the pole's velocity. The potential actions are simply: move the cart left or move the cart right.

For more information on Q Learning, please see: https://en.wikipedia.org/wiki/Q-learning

Epsilon Greedy: 

Binning:

