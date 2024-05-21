from random import choices
import numpy as np
from scipy.special import softmax

#ps_pos = np.array([
#    [4, 2, 1, 2, 3, 7],
#    [12, 14, 17, 11, 12, 13]
#])
#ps_orientation = [0.45, 0.49, 0.57, 0.43, 0.44, 0.47]
#weight = [3, 9, 0.5, 2, 7, 8]      


def resampling(ps_pos, ps_orientation, ps_weights):

    index = [i for i in range(len(ps_orientation))]
    Num_particles = len(index)
    sum_weights = sum(ps_weights)

    indices_resamp = choices(index, ps_weights, k = Num_particles)
    ps_pos = np.transpose(np.array([ps_pos[:,int(n)]for n in indices_resamp]))
    ps_orientation = [ps_orientation[int(n)]for n in indices_resamp]
    ps_weights = [1/Num_particles]*Num_particles

    return ps_pos, ps_orientation, ps_weights

#ps_pos, ps_orientation, ps_weights = resampling(ps_pos, ps_orientation, weight)

#print(ps_pos)
#print(ps_orientation)
#print(weight)



