import numpy as np
from random import choices
import matplotlib.pyplot as plt

# Initialize state
def initialize_state(x=0, y=0, theta=0):
    return np.array([x, y, theta])

# Generate control sequence for a square pattern
def generate_control_sequence_square_pattern(n_steps):
    control_sequence = []
    for i in range(n_steps):
        if i < n_steps // 4:
            control_sequence.append((1.0, 0.0))  # Move straight
        elif i < n_steps // 2:
            control_sequence.append((1.0, np.pi / 2 / (n_steps // 4)))  # Turn 90 degrees
        elif i < 3 * n_steps // 4:
            control_sequence.append((1.0, 0.0))  # Move straight
        else:
            control_sequence.append((1.0, np.pi / 2 / (n_steps // 4)))  # Turn 90 degrees
    return control_sequence

# Transition model for particle movement
def transition_model(X, U, dt=1.0, process_noise_std=[0.1, 0.1, 0.01], bounds=[0, 15, 0, 15]):
    x, y, theta = X
    v, omega = U
    process_noise = np.random.normal(0, process_noise_std)
    x += (v * np.cos(theta) * dt) + process_noise[0]
    y += (v * np.sin(theta) * dt) + process_noise[1]
    theta += (omega * dt) + process_noise[2]
    
    # Ensure the particles remain within the bounds
    x = np.clip(x, bounds[0], bounds[1])
    y = np.clip(y, bounds[2], bounds[3])
    
    return np.array([x, y, theta])

# Observation model
def observation_model(X, landmark_pos, measurement_noise_std=0.1):
    x, y, theta = X
    lx, ly = landmark_pos
    distance = np.sqrt((lx - x)**2 + (ly - y)**2)
    distance += np.random.normal(0, measurement_noise_std)
    return distance

# Generate synthetic data
def generate_data(X_0, U, landmark_pos, n_steps, dt=1.0, bounds=[0, 15, 0, 15]):
    X = X_0
    X_sequence = [X_0]
    Z_sequence = [observation_model(X_0, landmark_pos)]
    
    for t in range(n_steps):
        X = transition_model(X, U[t], dt, bounds=bounds)
        Z = observation_model(X, landmark_pos)
        X_sequence.append(X)
        Z_sequence.append(Z)
    
    return np.array(X_sequence), np.array(Z_sequence)

# Sample motion model for particle movement
def sample_motion_model(n, alpha, yaw, d, x_variables, y_variables):
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

# Prediction step for particle movement
def predictionStep(ps_pos, ps_orientation, dist, bearing, new_orientation, process_noise_std=[0.1, 0.1, 0.01], bounds=[0, 15, 0, 15]):
    n = ps_pos.shape[0]
    d = dist + np.random.normal(0, process_noise_std[0], n)
    alpha = bearing + np.random.normal(0, process_noise_std[1], n)
    yaw = new_orientation + np.random.normal(0, process_noise_std[2], n)
    
    new_positions = sample_motion_model(n, alpha, yaw, d, ps_pos[:, 0], ps_pos[:, 1])
    
    # Reflect particles at the boundaries
    for i in range(n):
        if new_positions[i, 0] < bounds[0] or new_positions[i, 0] > bounds[1]:
            new_positions[i, 0] = np.clip(new_positions[i, 0], bounds[0], bounds[1])
            ps_orientation[i] = np.pi - ps_orientation[i] + np.random.normal(0, process_noise_std[2])
        if new_positions[i, 1] < bounds[2] or new_positions[i, 1] > bounds[3]:
            new_positions[i, 1] = np.clip(new_positions[i, 1], bounds[2], bounds[3])
            ps_orientation[i] = -ps_orientation[i] + np.random.normal(0, process_noise_std[2])
    
    ps_pos[:, 0] = new_positions[:, 0]
    ps_pos[:, 1] = new_positions[:, 1]
    
    ps_orientation = (ps_orientation + yaw.reshape(-1, 1)) % (2 * np.pi)
    ps_orientation = (ps_orientation + np.pi) % (2 * np.pi) - np.pi

    return ps_pos, ps_orientation

def getParticleRanges(ps_pos, ps_orientation, pixel_matrix):
    n_particles = ps_pos.shape[0]
    n_scans = 180  # Example number of lidar scans
    simulated_scans = np.zeros((n_particles, n_scans))  # Placeholder for actual implementation
    
    for i in range(n_particles):
        x, y, theta = ps_pos[i]  # Ensure correct unpacking
        for j in range(n_scans):
            angle = theta + (j - n_scans // 2) * np.pi / n_scans  # Assuming 180-degree FOV
            scan_range = np.inf
            for r in np.arange(0, 5, 0.1):  # Assuming max range of 5 meters
                dx = x + r * np.cos(angle)
                dy = y + r * np.sin(angle)
                if int(dx.item()) >= pixel_matrix.shape[0] or int(dy.item()) >= pixel_matrix.shape[1] or pixel_matrix[int(dx.item()), int(dy.item())] == 0:
                    scan_range = r
                    break
            simulated_scans[i, j] = scan_range
    return simulated_scans


# Update weights based on sensor data
def updateWeights(ps_pos, ps_orientation, scan_data, pixel_matrix, map_metadata):
    simulated_scans = getParticleRanges(ps_pos, ps_orientation, pixel_matrix)
    n_particles = ps_pos.shape[0]
    
    weights = np.zeros(n_particles)
    sigma = 0.2  # Standard deviation for the Gaussian model
    for i in range(n_particles):
        diff = scan_data - simulated_scans[i]
        likelihood = np.exp(-0.5 * np.sum(diff**2) / sigma**2)
        weights[i] = likelihood
    
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        weights = np.ones(n_particles) / n_particles
    else:
        weights /= weight_sum
    
    return weights.reshape(-1, 1)

# Resampling step
def resampling(ps_pos, ps_orientation, ps_weights):
    index = np.arange(len(ps_orientation))
    num_particles = len(index)
    
    ps_weights = ps_weights / np.sum(ps_weights)
    
    indices_resampled = choices(index, ps_weights.flatten(), k=num_particles)
    
    ps_pos_resampled = ps_pos[indices_resampled]
    ps_orientation_resampled = ps_orientation[indices_resampled]
    ps_weights_resampled = np.ones(num_particles) / num_particles
    
    return ps_pos_resampled, ps_orientation_resampled, ps_weights_resampled

# Visualize particles, true position, and landmark
def visualize_particles(ps_pos, true_pos, landmark_pos, step, lidar_data):
    plt.clf()
    plt.scatter(true_pos[0], true_pos[1], c='r', s=100, marker='x', label='True Position')
    plt.scatter(ps_pos[:, 0], ps_pos[:, 1], c='b', s=10, label='Particles')
    plt.scatter(landmark_pos[0], landmark_pos[1], c='g', s=100, marker='o', label='Landmark')
    plt.scatter(lidar_data[:, 0], lidar_data[:, 1], c='m', s=50, marker='.', label='LiDAR Data')
    plt.plot([0, 15, 15, 0, 0], [0, 0, 15, 15, 0], 'k-', lw=2)  # Room boundaries
    plt.xlim(-1, 19)
    plt.ylim(-1, 19)
    plt.legend()
    plt.title(f'Step {step}')
    plt.pause(0.01)


# Initialization
X_0 = initialize_state()
U = generate_control_sequence_square_pattern(25)
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
    v, omega = U[t]  # Use the control sequence values
    dist = v
    bearing = omega
    new_orientation = 0.01  # This can be adjusted as needed
    ps_pos, ps_orientation = predictionStep(ps_pos, ps_orientation, dist, bearing, new_orientation)
    
    # Update weights
    scan_data = Z_sequence[t]  # Using synthetic scan data
    ps_weights = updateWeights(ps_pos, ps_orientation, scan_data, np.zeros((20, 20)), None)
    
    # Resampling
    ps_pos, ps_orientation, ps_weights = resampling(ps_pos, ps_orientation, ps_weights)
    
    # Simulate LiDAR data for visualization
    simulated_lidar = getParticleRanges(np.array([[X_sequence[t][0], X_sequence[t][1], 0]]), np.array([X_sequence[t][2]]), np.zeros((20, 20)))
    lidar_points = np.array([[X_sequence[t][0] + r * np.cos(theta), X_sequence[t][1] + r * np.sin(theta)] 
                             for r, theta in zip(simulated_lidar[0], np.linspace(-np.pi, np.pi, len(simulated_lidar[0])))])
    
    # Visualize particles every 5 steps
    if t % 5 == 0:
        visualize_particles(ps_pos, X_sequence[t], landmark_pos, t, lidar_points)
    
    # Print the state for debugging every 5 steps
    if t % 5 == 0:
        print(f'Step {t}, State: {ps_pos.mean(axis=0)}, Orientation: {ps_orientation.mean()}, Weight: {ps_weights.mean()}')



plt.show()
