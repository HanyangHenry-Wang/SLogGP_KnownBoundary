import numpy as np
import gymnasium as gym
import torch

def heuristic_Controller(s, w): # I think s is the state and w is the parameter that I need to tune
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


class lunar12:
 

    def __init__(self, seed):
        self.dim = 12
        self.seed =seed

        
        self.bounds = torch.tensor([[[0.,2.]]*self.dim ]).reshape(self.dim ,2).T
            


    def __call__(self, x, plotting_args=None):
        
        x = x.numpy().reshape(12,)
        

        env = gym.make("LunarLander-v2", render_mode= None) #"human"

        observation, info = env.reset(seed=0)

        action = 0
        observation, reward, terminated, truncated, info = env.step(action)


        reward_sum = reward

        for _ in range(200):
     
            action = heuristic_Controller(s=observation, w=x)
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            reward_sum += reward
            #print(reward)

            if terminated or truncated:
                #print('this is the final reward: ',reward)
                reward_sum += reward
                
                break
                # observation, info = env.reset()

        env.close()
        
        return -torch.tensor(reward_sum.ravel())



