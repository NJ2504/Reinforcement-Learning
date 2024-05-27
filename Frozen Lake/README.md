This project is an implementation of Frozen Lakes Game. The environment setup is as follows: The enitre grid size is (5,5). 
The start point is defined at (0,0) and the goal state is defined at (4,4). There are also holes defined at (1,0), (1,3), (3,1), (4,2). 
If the agent falls into the hole there is a negative reward assigned to it. 
There are only four possible actions and they are: "Up", "Down", "Left", "Right".
The reward for the goal state is 10.0, the reward for the hole state is -5.0, the reward for every other state is -1.0 
Different variations of Hyperparameter settings are been used and also epsilon decay is also implemented. 
The code also provides visualition and Q-Table of the learning to understand the stability of the learning done by the agent.
