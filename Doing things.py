# Simple script to demonstrate how to use the environment as a black box.

# Import the environment
import time

# other dependencies
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import envV2 as ENV
#Import agent + policy model : Reinforce
from Reinforce_keras import Agent

# hyperparameters
learning_rate = 3e-4

# Constants
gamma = 0.99
n_steps = 30
n_episodes = 100
num_inputs = 10000
num_actions = 4
filename = "reinforce_7L_Weights_final_test1"
#Load the environment, it has a number of variables that can be initailized.
#Here we just set the movement speed of the drone and drone size radius.
env = ENV.sonarEnv(rotationAngle=90)
agent = Agent(ALPHA=learning_rate, 
              GAMMA=gamma, 
              input_dims=num_inputs, 
              n_actions=num_actions,
              load = False, 
              fname_policy=str("models/"+filename+"_policy.h5"))


#steps record
all_lengths = []
average_lengths = []
#reward record
all_rewards = []

for i in range(n_episodes):
    done=False
    steps = 0
    rewards = []
    state = env.reset()
    agent.reset_ep_graph()
    while not done and steps < n_steps:
        #select action
        action = agent.choose_action(state)
        #step
        new_state, reward, done, info = env.step(action)
        rewards.append(reward)
        #store transitions: state, action, reward, self.log_prob in self.log_probs, self.log_prob in self.log_probs
        steps += 1
        agent.store_transition(state,action, rewards, steps)
        state = new_state
        #if(i%500 == 0):
            #env.render(steps,i,rewards[-1])      
    all_rewards.append(np.sum(rewards))
    all_lengths.append(steps)
    average_lengths.append(np.mean(all_lengths[-10:]))
    agent.learn()
    if(i%10== 0):
        agent.print_ep_stats(i, all_rewards)
    if (i%500 == 0):#every 500 ep
        agent.save_model()

# transfer function to agnent class
agent.plot_results(filename, all_rewards, all_lengths, average_lengths)
# Frees some memory when finished with the environment
env.close()
