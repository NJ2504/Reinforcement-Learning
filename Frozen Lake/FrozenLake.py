import numpy as np
import matplotlib.pyplot as plt

# Initializing the FrozenLake environment 
class FrozenLake:
    def __init__(self):
        # Initializing the Grid Specifications
        self.grid_size = (5, 5)
        self.start_state = (0, 0)
        self.goal_state = (4, 4)
        self.holes = [(1, 0), (1, 3), (3, 1), (4, 2)]
        self.state = self.start_state
        self.actions = ['up', 'down', 'left', 'right']
        self.rewards = {'goal': 10.0, 'hole': -5.0, 'default': -1.0}

    def reset(self):
        # Initialise the reset of the environemnt to the start_state
        # It is crucial to start the new episode from the start_state
        self.state = self.start_state
        return self.state

    def step(self, action):
        # Calculate the next_state
        next_state = self.get_next_state(self.state, action)
        # Based upon the next_state calculate the associated reward
        reward = self.get_reward(next_state)
        # Check whether the state is a goal_state or a hole
        done = next_state == self.goal_state or next_state in self.holes
        # If the episode is not done then the agent's state is updated to the next one.
        # Else if the episode is done, the state is reset to start_state
        self.state = next_state if not done else self.start_state
        return next_state, reward, done

    def get_next_state(self, state, action):
        # Taking in the current state, the actions 'up', 'down', 'left' and 'right' is defined.
        if action == 'up':
            next_state = (max(0, state[0] - 1), state[1])
        elif action == 'down':
            next_state = (min(self.grid_size[0] - 1, state[0] + 1), state[1])
        elif action == 'left':
            next_state = (state[0], max(0, state[1] - 1))
        elif action == 'right':
            next_state = (state[0], min(self.grid_size[1] - 1, state[1] + 1))
        else:
            next_state = state
        return next_state

    def get_reward(self, state):
        # Defines what reward the agent will get when it lands on the new state.
        if state == self.goal_state:
            return self.rewards['goal']
        elif state in self.holes:
            return self.rewards['hole']
        else:
            return self.rewards['default']
        
env = FrozenLake()

# Initialize the Q-Learning Algorithm
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=0.1,epsilon_decay=None,decay_factor=0.01):
        self.env = env
        self.q_table = np.zeros((*env.grid_size, len(env.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.episode_rewards = []
        self.epsilon_decay=epsilon_decay
        self.decay_factor=decay_factor

    def choose_action(self, state):
        # Generate a random number if the number is lesser than epsilon then the agent will explore
        if np.random.rand() < self.epsilon:
            # During exploration, let the agent select an action
            return np.random.choice(self.env.actions)
        else:
            # Else select the action with the hightest Q-value for the current state
            return self.env.actions[np.argmax(self.q_table[state])]

    def update_epsilon(self, episode, total_episodes):
        if self.epsilon_decay:
            if episode == 0:
                self.epsilon = self.initial_epsilon
            if episode % 20 == 0:
                self.epsilon = self.epsilon - self.epsilon*self.decay_factor

    def learn(self, total_episodes=10000):
        # Episodes is set to 10000
        # Loop through each and every episode
        for episode in range(total_episodes):
            # Reset the environment
            state = self.env.reset()
            # Initialize a flag to represent whether current episode ended or not
            done = False
            # Initializing to store cumulative reward
            total_reward = 0
            # Change epsilon value based upon the epsilon decay formulation. 
            self.update_epsilon(episode, total_episodes)
            # Running the loop until the episode isn't finished
            while not done:
                # Selecting an action from the current state
                action_index = self.env.actions.index(self.choose_action(state))
                # Perform action, go to the next state and receive the reward and check whether the episode is done
                next_state, reward, done = self.env.step(self.env.actions[action_index])
                # Find the maximum Q-value for the next_state
                next_max = np.max(self.q_table[next_state])
                # Calculate Q-value for the current state and action pair.
                self.q_table[state + (action_index,)] = (1 - self.alpha) * self.q_table[state + (action_index,)] + \
                                                        self.alpha * (reward + self.gamma * next_max)
                # Update current state to next_state
                state = next_state
                # Accumulate all the rewards
                total_reward += reward
            self.episode_rewards.append(total_reward)
        return self.q_table, self.episode_rewards
# Initializing Basic Hyperparameters alpha = 0.5, gamma = 0.9, epsilon = 0.1
q_learning_basic = QLearning(env)
# Learning the policy 
q_table_basic, rewards_basic = q_learning_basic.learn()
# Plotting the learning curve
plt.plot(rewards_basic)
plt.title('Learning Curve for Basic Parameter Settings')
plt.xlabel('Episodes')
plt.ylabel('Rewards per Episode')
plt.show()


# Custom Hyperparameter settings
alpha = 0.6
gamma = 0.8  
epsilon = 0.2  
# Defining epsilon decay by, ensuring that the calculated value doesn't fall below 0.01
# Made this to ensure the agent does some extent of exploration 
epsilon_decay_custom = lambda episode, total_episodes: max(0.01, 0.2 - (0.2 - 0.01) * (episode / total_episodes))
# Initializing Q-Learning
q_learning_decay = QLearning(env, alpha= alpha, gamma= gamma, epsilon= epsilon, epsilon_decay= 1)
# Learning the policy
q_table_decay, rewards_decay = q_learning_decay.learn()
# Plotting the Learning Curve
plt.plot(rewards_decay)
plt.title('Learning Curve for Epsilon Decay')
plt.xlabel('Episodes')
plt.ylabel('Rewards per Episode')
plt.show()

# Custom Hyperparameter settings
alpha = 0.6
gamma = 0.8  
epsilon = 0.2  
epsilon_decay_custom = lambda episode, total_episodes: max(0.01, 0.2 - (0.2 - 0.01) * (episode / total_episodes))
# Initializing Q-Learning
q_learning_custom = QLearning(env, alpha= alpha, gamma= gamma, epsilon= epsilon, epsilon_decay= 1,decay_factor=0.1)
# Learning the policy
q_table_custom, rewards_custom = q_learning_custom.learn()
# Plotting the Learning Curve
plt.plot(rewards_custom)
plt.title('Learning Curve for Custom Hyperparameters')
plt.xlabel('Episodes')
plt.ylabel('Rewards per Episode')
plt.show()

# Pringting out the Q-Tables
def print_q_table_corrected(q_table):
    max_q_values = np.max(q_table, axis=2)
    print(max_q_values)
print("Q-Table for Basic Hyperparameters:")
print_q_table_corrected(q_table_basic)
print("\nQ-Table for Epsilon Decay:")
print_q_table_corrected(q_table_decay)
print("\nQ-Table for Custom Hyperparameters:")
print_q_table_corrected(q_table_custom)

