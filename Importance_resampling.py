from random import choices
import numpy as np

index = [0, 1, 2, 3, 4, 5]
x = [4, 2, 1, 2, 3, 7]
y = [12, 14, 17, 11, 12, 13]
theta = [0.45, 0.49, 0.57, 0.43, 0.44, 0.47]
weight = [3, 9, 0.5, 2, 7, 8]      

particles = np.array([
    index,
    x,
    y,
    theta,
    weight,
])

def importance_resampling(particles):
   
    Num_particles = len(particles[0,:])
    sum_weights = sum(particles[4,:])
    normalized_weights = [w / sum_weights for w in particles[4,:]]

    indices_resamp = choices(particles[0,:], particles[4,:], k = Num_particles)
    particles_resamp = np.transpose(np.array([particles[:,int(n)]for n in indices_resamp]))

    particles_resamp[4,:] = 1/Num_particles

    return particles_resamp

resampled_particles = importance_resampling(particles)
print(resampled_particles)




