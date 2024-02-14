import gym
import numpy as np
import random
import time
from IPython.display import clear_output

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')


iteration = 10 #set 10 iteration
av_total_rewards = []
tot_num_actions = []


 # Training parameters for Q learning
alpha = 0.9  # Learning rate
gamma = 0.9  # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500  # per each episode


# Q table for rewards
Q_reward = np.zeros((env.observation_space.n, env.action_space.n))

    #av_total_rewards = 0
    #tot_num_actions = 0

# Training with Q-learning
for episode in range(num_of_episodes):
    state = env.reset()[0]
    for _ in range(num_of_steps):
            
        action = np.argmax(Q_reward[state, :])

        # Take the chosen action
        new_state, reward, done, truncated, info = env.step(action)

        # Update the Q-table using Q-learning equation
        Q_learning_value = (1 - alpha) * Q_reward[state, action] + alpha * (
            reward + gamma * np.max(Q_reward[new_state, :])
        )

        Q_reward[state, action] = Q_learning_value 

        state = new_state
        #num_actions += 1

            #if done:
                #total_rewards += reward
                #break

    #average_total_rewards.append(total_rewards)
    #average_num_actions.append(num_actions)

    # Close the environment
    #env.close()
# Testing
for run in range(iteration):
    
    state = env.reset()[0]
    total_reward = 0
    num_actions = 0
    for t in range(50):
        action = np.argmax(Q_reward[state, :])
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        num_actions += 1
        print(env.render())
        time.sleep(1)
        if done:
            print("##########################")
            print("Total reward for iteration %d: %d" % (run + 1, total_reward))
            print("##########################")
            av_total_rewards.append(total_reward)
            tot_num_actions.append(num_actions)
            break

    

    # Close the environment
    env.close()

# Calculate the average total reward and average number of actions across runs
avg_total_reward = np.mean(av_total_rewards)
avg_num_actions = np.mean(tot_num_actions)

print("Average Total Reward:", avg_total_reward)
print("Average Number of Actions:", avg_num_actions)



