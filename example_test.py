import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    m,n = cmap.shape
    belief = 1/np.size(cmap) * np.ones([m,n])
    #### Test your code here
    for t in range(30):
        belief,belief_state = histogram_filter(cmap, belief, actions[t], observations[t])
    
    
    '''
    m,n = cmap.shape
    belief = 1/np.size(cmap) * np.ones([m,n])
    history = np.zeros([30,2])
    eta = 0
    p_sensor = np.zeros(belief.shape)
    for t in range(30):  
        
        u = actions[t]
        new_belief = 0
        for (r,c),prob in np.ndenumerate(belief):
            x = c
            y = m-1-r
            p_action = np.zeros(belief.shape)
            if (0 <= x + u[0] <= n-1) & (0 <= y + u[1] <= m-1):
                p_action[r,c] = 0.1
                p_action[r - u[1],c + u[0]] = 0.9
            else:
                p_action[r,c] = 1
            new_belief += prob * p_action
        belief = new_belief
        
        z = observations[t]
        p_sensor[np.where(cmap == z)] = 0.9
        p_sensor[np.where(cmap != z)] = 0.1
        belief = p_sensor * belief

        eta = np.sum(belief)
        belief = belief/eta

        belief_state = np.where(belief==np.max(belief))
        history[t] = np.array([belief_state[1][0],19-belief_state[0][0]])
        '''