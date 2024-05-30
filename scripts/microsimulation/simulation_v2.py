import numpy as np
from random import choices
import matplotlib.pyplot as plt

def initialize_state(x=0, y=0, theta=0):
    """Initialize the robot state."""
    return np.array([x, y, theta])

def generate_control_sequence(n_steps, v=1.0, omega=0.1):
    """Generate a sequence of control inputs."""
    return [(v, omega) for _ in range(n_steps)]

def transition_model(X, U, dt=1.0, process_noise_std=[0.1, 0.1, 0.01], bounds=[0, 20, 0, 20]):
    """Transition model with process noise."""
    x, y, theta = X
    v, omega = U
    process_noise = np.random.normal(0, process_noise_std)
    x += (v * np.cos(theta) * dt) + process_noise[0]
    y += (v * np.sin(theta) * dt) + process_noise[1]
    theta += (omega * dt) + process_noise[2]
    
    # Reflect particles at the boundaries
    if x < bounds[0] or x > bounds[1]:
        x = np.clip(x, bounds[0], bounds[1])
        theta = np.pi - theta
    if y < bounds[2] or y > bounds[3]:
        y = np.clip(y, bounds[2], bounds[3])
        theta = -theta
    
    return np.array([x, y, theta])

def observation_model(X, landmark_pos, measurement_noise_std=0.1):
    """Observation model with measurement noise."""
    x, y, theta = X
    lx, ly = landmark_pos
    distance = np.sqrt((lx - x)**2 + (ly - y)**2)
    distance += np.random.normal(0, measurement_noise_std)
    return distance

def generate_data(X_0, U, landmark_pos, n_steps, dt=1.0, bounds=[0, 20, 0, 20]):
    """Generate synthetic state and observation data."""
    X = X_0
    X_sequence = [X_0]
    Z_sequence = [observation_model(X_0, landmark_pos)]
    
    for t in range(n_steps):
        X = transition_model(X, U[t], dt, bounds=bounds)
        Z = observation_model(X, landmark_pos)
        X_sequence.append(X)
        Z_sequence.append(Z)
    
    return np.array(X_sequence), np.array(Z_sequence)

def sample_motion_model(n, alpha, yaw, d, x_variables, y_variables):
    """Sample motion model for particle filter."""
    def sum_matrices(n, alpha, yaw):
        matrix_alpha = np.array(alpha).reshape(n, 1)
        matrix_yaw = np.array(yaw).reshape(n, 1)
        result_matrix = matrix_alpha + matrix_yaw
        return result_matrix

    def calculate_trig_functions(matrix):
        cos_matrix = np.cos(matrix)
        sin_matrix = np.sin(matrix)
        return cos_matrix, sin_matrix

    def multiply_with_d(cos_matrix, sin_matrix, d):
        matrix = np.vstack((cos_matrix.T, sin_matrix.T)).T
        result_matrix = d.reshape(n, 1) * matrix
        return result_matrix

    def sum_with_variables(x_variables, y_variables, final_result):
        variables_matrix = np.vstack((x_variables, y_variables)).T
        result_matrix = final_result + variables_matrix
        return result_matrix

    result = sum_matrices(n, alpha, yaw)
    cos_result, sin_result = calculate_trig_functions(result)
    final_result = multiply_with_d(cos_result, sin_result, d)
    final_sum = sum_with_variables(x_variables, y_variables, final_result)
    return final_sum

def predictionStep(ps_pos, ps_orientation, dist, bearing, new_orientation, process_noise_std=[0.1, 0.1, 0.01], bounds=[0, 20, 0, 20]):
    """Predict the next state of particles."""
    n = ps_pos.shape[0]
    d = dist + np.random.normal(0, process_noise_std[0], n)
    alpha = bearing + np.random.normal(0, process_noise_std[1], n)
    yaw = new_orientation + np.random.normal(0, process_noise_std[2], n)
    
    new_positions = sample_motion_model(n, alpha, yaw, d, ps_pos[:, 0], ps_pos[:, 1])
    
    # Reflect particles at the boundaries
    new_positions[:, 0] = np.clip(new_positions[:, 0], bounds[0], bounds[1])
    new_positions[:, 1] = np.clip(new_positions[:, 1], bounds[2], bounds[3])
    
    for i in range(n):
        if new_positions[i, 0] == bounds[0] or new_positions[i, 0] == bounds[1]:
            yaw[i] = np.pi - yaw[i]
        if new_positions[i, 1] == bounds[2] or new_positions[i, 1] == bounds[3]:
            yaw[i] = -yaw[i]
    
    ps_pos[:, 0] = new_positions[:, 0]
    ps_pos[:, 1] = new_positions[:, 1]
    
    ps_orientation = (ps_orientation + yaw.reshape(-1, 1)) % (2 * np.pi)
    ps_orientation = (ps_orientation + np.pi) % (2 * np.pi) - np.pi

    return ps_pos, ps_orientation

def getParticleRanges(ps_pos, ps_orientation, pixel_matrix):
    """Simulate lidar measurements for each particle."""
    n_particles = ps_pos.shape[0]
    n_scans = 180  # Example number of lidar scans
    simulated_scans = np.zeros((n_particles, n_scans))  # Placeholder for actual implementation
    
    # Simulate scan ranges for each particle
    for i in range(n_particles):
        x, y = ps_pos[i]
        theta = ps_orientation[i]
        for j in range(n_scans):
            angle = theta + (j - n_scans // 2) * np.pi / n_scans  # Assuming 180-degree FOV
            scan_range = np.inf
            # Simulate LIDAR scan (simple straight-line intersection with the map)
            for r in np.arange(0, 5, 0.1):  # Assuming max range of 5 meters
                dx = x + r * np.cos(angle)
                dy = y + r * np.sin(angle)
                if int(dx[0]) >= pixel_matrix.shape[0] or int(dy[0]) >= pixel_matrix.shape[1] or pixel_matrix[int(dx[0]), int(dy[0])] == 0:
                    scan_range = r
                    break
            simulated_scans[i, j] = scan_range
    return simulated_scans

def updateWeights(ps_pos, ps_orientation, scan_data, pixel_matrix, map_metadata):
    """Update weights of particles based on the observation model."""
    simulated_scans = getParticleRanges(ps_pos, ps_orientation, pixel_matrix)
    n_particles = ps_pos.shape[0]
    
    # Calculate the likelihood of the observed scan given the particle's state
    weights = np.zeros(n_particles)
    sigma = 0.2  # Standard deviation for the Gaussian model
    for i in range(n_particles):
        # Compare the actual scan with the simulated scan for this particle
        diff = scan_data - simulated_scans[i]
        likelihood = np.exp(-0.5 * np.sum(diff**2) / sigma**2)
        weights[i] = likelihood
    
    # Normalize the weights
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        weights = np.ones(n_particles) / n_particles
    else:
        weights /= weight_sum
    
    return weights.reshape(-1, 1)

def resampling(ps_pos, ps_orientation, ps_weights):
    """Resample particles based on their weights."""
    index = np.arange(len(ps_orientation))
    num_particles = len(index)
    
    # Normalize weights
    ps_weights = ps_weights / np.sum(ps_weights)
    
    # Resample indices based on weights
    indices_resampled = choices(index, ps_weights.flatten(), k=num_particles)
    
    # Create new resampled particles
    ps_pos_resampled = ps_pos[indices_resampled]
    ps_orientation_resampled = ps_orientation[indices_resampled]
    ps_weights_resampled = np.ones(num_particles) / num_particles
    
    return ps_pos_resampled, ps_orientation_resampled, ps_weights_resampled

def visualize_particles(ps_pos, true_pos, landmark_pos, step):
    """Visualize particles, true position, and landmark."""
    plt.clf()
    plt.scatter(true_pos[0], true_pos[1], c='r', s=100, marker='x', label='True Position')  # Plot true position first
    plt.scatter(ps_pos[:, 0], ps_pos[:, 1], c='b', s=10, label='Particles')  # Plot particles on top
    plt.scatter(landmark_pos[0], landmark_pos[1], c='g', s=100, marker='o', label='Landmark')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()
    plt.title(f'Step {step}')
    plt.pause(0.01)

# Initialization
X_0 = initialize_state()
U = generate_control_sequence(25)
landmark_pos = (10, 10)
ps_pos = np.random.rand(100, 2) * 20  # Initialize particles randomly within a 20x20 area
ps_orientation = np.random.rand(100, 1) * 2 * np.pi - np.pi  # Random orientations
ps_weights = np.ones((100, 1)) / 100

# Generate synthetic data
X_sequence, Z_sequence = generate_data(X_0, U, landmark_pos, 25)

# Set up the plot
plt.figure()

# Main loop for the micro-simulation
for t in range(25):
    dist, bearing, new_orientation = (1, 0.1, 0.01)  # Example values
    ps_pos, ps_orientation = predictionStep(ps_pos, ps_orientation, dist, bearing, new_orientation)
    
    # Update weights
    scan_data = Z_sequence[t]  # Using synthetic scan data
    ps_weights = updateWeights(ps_pos, ps_orientation, scan_data, np.zeros((20, 20)), None)
    
    # Resampling
    ps_pos, ps_orientation, ps_weights = resampling(ps_pos, ps_orientation, ps_weights)
    
    # Visualize particles every 5 steps
    if t % 5 == 0:
        visualize_particles(ps_pos, X_sequence[t], landmark_pos, t)
    
    # Print the state for debugging every 5 steps
    if t % 5 == 0:
        print(f'Step {t}, State: {ps_pos.mean(axis=0)}, Orientation: {ps_orientation.mean()}, Weight: {ps_weights.mean()}')

plt.show()
