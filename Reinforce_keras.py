# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:31:48 2020

@author: sefun
    todo: 
    sum and averaging rewards

    agent and envirments
    summing for each episode


"""
import tensorflow as tf
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import activations
from tensorflow.keras.layers import (Activation, Dense, Dropout, Flatten,
                                     Input, concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99,n_actions=4, input_dims=128,load = False,
                 fname_policy=None):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [0, 1, 2, 3]
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        #ep plot data record
        self.action_Dict = {
            0 : 'U',
            1 : 'D',
            2 : 'L',
            3 : 'R'
        }
        self.stepDirct_history = [0,0,0,0]
        self.steps_arr = [0]
        self.score_arr = [0]
        self.action_arr = ['U']
        
        #load model and weights
        self.load = load
        self.model_file_policy = str(fname_policy)
        self.policy, self.predict = self.build_policy_network()
        #load weights
        if (self.model_file_policy != None and self.load == True):
            self.policy.load_weights(self.model_file_policy)
        
        
    def build_policy_network(self):
        #env2d = Input(shape=(self.input_dims,self.input_dims))

        '''
        7 layer dense sigmoid relu sigmoid 
        '''
        #input layer
        env1d = Input(name="state",shape=(self.input_dims))
        env = Flatten()(env1d)
        #1st layer
        denseSigmoid1 = Dense(64, activation=activations.sigmoid)(env)
        dropout1 = Dropout(.3, input_shape=(64,))(denseSigmoid1)
        #2nd layer
        denseRelu2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(.3, input_shape=(64,))(denseRelu2)
        #3rd layer
        denseSigmoid3 = Dense(64, activation=activations.sigmoid)(dropout2)
        dropout3 = Dropout(.3, input_shape=(64,))(denseSigmoid3)
        #4th layer
        denseRelu4 = Dense(32, activation='relu')(dropout3)
        dropout4 = Dropout(.3, input_shape=(32,))(denseRelu4)
        #5th layer
        denseSigmoid5 = Dense(16, activation=activations.sigmoid)(dropout4)
        dropout5 = Dropout(.3, input_shape=(16,))(denseSigmoid5)
        #6th layer
        denseRelu6 = Dense(16, activation='relu')(dropout5)
        dropout6 = Dropout(.3, input_shape=(16,))(denseRelu6)
        #7th layer
        denseSigmoid7 = Dense(8, activation=activations.sigmoid)(dropout6)
        #output layer
        probs = Dense(self.n_actions, activation='softmax')(denseSigmoid7)
       
        
        advantages = Input(name="advantage",shape=[1])
        def custom_loss(y_pred, y_true):
            out = K.clip(y_pred, 1e-8,  1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*advantages)
        
        policy = Model(inputs=[env1d,advantages], outputs=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)
        
        self.predict = Model(inputs=[env1d], outputs=[probs])
        
        return policy, self.predict
    
    def choose_action(self, observation):
        #state = observation[np.newaxis, :]
        #probabilities = self.predict.predict(state)
        #action = np.random.choice(a = self.action_space, p=probabilities[0])
        action = np.random.choice(a = self.action_space)
        return action
    
    def store_transition(self, observation, action, rewards, steps):
        state = observation
        self.stepDirct_history[action] += 1
        self.action_memory.append(action)
        self.state_memory.append(state)
        self.reward_memory.append(rewards[-1])
        
        self.steps_arr.append(steps)
        self.score_arr.append(np.sum(rewards))
        self.action_arr.append(self.action_Dict[action])
        
    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        #
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]*discount
                discount *= self.gamma
                
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean)/std
        #
        cost = self.policy.train_on_batch([state_memory ,self.G], action_memory)
        #
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
    def save_model(self):
        self.policy.save_weights(self.model_file_policy)
        
    def print_ep_stats(self, ep, all_rewards):
        print('ep: %6.1d' % ep, ' score : %6.1f' % all_rewards[-1],' steps : %-6.1f' % (self.steps_arr[-1]-1),
          'average_score : %5.2f' % np.mean(all_rewards[-10:]),
          'F:%4.1f' % (self.stepDirct_history[0]/sum(self.stepDirct_history)),
          'B:%4.1f' % (self.stepDirct_history[1]/sum(self.stepDirct_history)),
          'L:%4.1f' % (self.stepDirct_history[2]/sum(self.stepDirct_history)),
          'R:%4.1f' %(self.stepDirct_history[3]/sum(self.stepDirct_history)))
        if(ep%100== 0):
            self.plot_ep_graph(ep, all_rewards)
    
    def reset_ep_graph(self):
        self.stepDirct_history = [0,0,0,0]
        self.steps_arr = [0]
        self.score_arr = [0]
        self.action_arr = ['U']
        
    def plot_ep_graph(self, ep, all_rewards):
        # Function to map the colors as a list from the input list of x variables
        def pltcolor(lst):
            cols=[]
            for l in lst:
                if l=='U':
                    cols.append(0)
                elif l=='L':
                    cols.append(1)
                elif l=='R':
                    cols.append(2)
                else:
                    cols.append(3)
            return cols
    
        colors = ["green", "blue", "cyan", "red"]
        colormap = matplotlib.colors.ListedColormap(colors)
        color_indices = pltcolor(self.action_arr)
    
        fig, ax = plt.subplots()
        ax.plot(self.steps_arr, self.score_arr,'--', color='black')
        ax.scatter(self.steps_arr, self.score_arr, c=color_indices, cmap=colormap)
        plt.title('Agent Score Vs Action Steps '+str(ep)+' avg_score:'+str(np.mean(all_rewards[-10:])), fontsize=14)
        plt.xlabel('Num. Steps', fontsize=14)
        plt.ylabel('Agent Score', fontsize=14)
        
        gre_patch = mpatches.Patch(color='green', label='Forward')
        red_patch = mpatches.Patch(color='red', label='Back')
        pin_patch = mpatches.Patch(color='cyan', label='Turn Right')
        blu_patch = mpatches.Patch(color='blue', label='Turn Left')
        
        plt.legend(handles=[gre_patch, red_patch, blu_patch, pin_patch])
        plt.grid(True)
        plt.savefig('outputs/ep_graph/ep_0_rot'+str(ep)+'.png',transparent=False)
        plt.show()
        plt.close()
        #direction count
        plt.bar([1,2,3,4], height = self.stepDirct_history)
        plt.xticks([1,2,3,4], ['Forwards','backwards','T Left', 'T Right'])
        plt.savefig('outputs/ep_graph/Action_Prob_Dist_'+str(ep)+'.png',transparent=False)
        plt.show()
        plt.close()
        
    def plot_results(self, filename, all_rewards, all_lengths, average_lengths):
        smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        plt.plot(all_rewards)
        plt.plot(smoothed_rewards)
        plt.plot()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('outputs/ep_graph/'+filename+'_Length_result.png',transparent=False)
        plt.show()

        plt.plot(all_lengths)
        plt.plot(average_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Episode length')
        plt.savefig('outputs/ep_graph/'+filename+'_Reward_result.png',transparent=False)
        plt.show()
        
  