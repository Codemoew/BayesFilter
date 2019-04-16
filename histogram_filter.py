import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
        '''
        
        ### Your Algorithm goes Below.
        m,n = cmap.shape

        p_sensor = np.zeros(belief.shape)
        new_belief = 0
        for (r,c),prob in np.ndenumerate(belief):
            x = c
            y = m-1-r
            p_action = np.zeros(belief.shape)
            if (0 <= x + action[0] <= n-1) & (0 <= y + action[1] <= m-1):
                p_action[r,c] = 0.1
                p_action[r - action[1],c + action[0]] = 0.9
            else:
                p_action[r,c] = 1
            new_belief += prob * p_action
        belief = new_belief
        
        p_sensor[np.where(cmap == observation)] = 0.9
        p_sensor[np.where(cmap != observation)] = 0.1
        belief = p_sensor * belief

        eta = np.sum(belief)
        belief = belief/eta

        index = np.where(belief==np.max(belief))
        
        belief_state = np.array([index[1][0],m-1-index[0][0]])        
       
        return belief,belief_state
        
        
       