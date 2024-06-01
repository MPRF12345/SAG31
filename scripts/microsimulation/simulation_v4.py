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

# Laser simulator class
class LaserSimulator:
    def __init__(self, pixel_matrix, map_metadata):
        self.laser_scanner = {
            'min_range': 0.02, # meters
            'max_range': 5.6, # meters
            'min_angle': -120 * np.pi / 180,
            'max_angle': 120 * np.pi / 180,
            'angle_increment': 3.352 * np.pi / 180
        }
        self.pixel_matrix = pixel_matrix
        self.map_metadata = map_metadata

        # get max number of pixels for range
        max_range_pixels = int(self.laser_scanner['max_range'] / self.map_metadata['resolution'])
        self.pixel_range = np.arange(1, max_range_pixels + 1)
        self.n_scans = int((self.laser_scanner['max_angle'] - self.laser_scanner['min_angle']) / self.laser_scanner['angle_increment'])
        self.scan_angles = np.linspace(self.laser_scanner['min_angle'], self.laser_scanner['max_angle'], self.n_scans)

    def getParticleRanges(self, ps_pos, ps_orientation):
        n_particles = ps_pos.shape[0]
        ranges = np.zeros((n_particles, self.n_scans))

        for i in range(n_particles):
            x, y, theta = ps_pos[i]

            orientations = theta + self.scan_angles
            orientations = np.mod(orientations + np.pi, 2 * np.pi) - np.pi

            pixel_positions = self._getPixelPositions(orientations, x, y).astype(int)

            height, width = self.pixel_matrix.shape
            pixel_positions[:, :, 0] = np.clip(pixel_positions[:, :, 0], 0, width - 1)
            pixel_positions[:, :, 1] = np.clip(pixel_positions[:, :, 1], 0, height - 1)

            # Updated to correctly identify obstacles
            for j in range(self.n_scans):
                for k in range(len(self.pixel_range)):
                    px, py = pixel_positions[j, k]
                    if self.pixel_matrix[py, px] == 0:
                        ranges[i, j] = np.sqrt((px - x) ** 2 + (py - y) ** 2) * self.map_metadata['resolution']
                        break
                    ranges[i, j] = self.laser_scanner['max_range']

        return ranges

    def _getPixelPositions(self, orientations, x, y):
        y = -y

        q1 = orientations[(orientations > -np.pi / 4) & (orientations < np.pi / 4)]
        q2 = orientations[(orientations > np.pi / 4) & (orientations < 3 * np.pi / 4)]
        q3 = orientations[(orientations > 3 * np.pi / 4) | (orientations < -3 * np.pi / 4)]
        q4 = orientations[(orientations > -3 * np.pi / 4) & (orientations < -np.pi / 4)]

        first_angle = orientations[0]
        if first_angle < - 3 * np.pi / 4:
            quadrants = [3, 4, 1, 2]
        elif first_angle < -np.pi / 4:
            quadrants = [4, 1, 2, 3]
        elif first_angle < np.pi / 4:
            quadrants = [1, 2, 3, 4]
        elif first_angle < 3 * np.pi / 4:
            quadrants = [2, 3, 4, 1]
        else:
            quadrants = [3, 4, 1, 2]

        q_pixel_positions = {}
        if q1.shape[0] == 0:
            q_pixel_positions[1] = np.zeros((0, self.pixel_range.shape[0], 2))
        else:
            x_pixel_increments = x + self.pixel_range
            xs = np.tile(x_pixel_increments, (q1.shape[0], 1))
            slopes = np.reshape(np.tan(q1), (q1.shape[0], 1))
            bs = y - (slopes * x)

            ys = np.round((xs * slopes) + bs, 0).astype(int)
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            q_pixel_positions[1] = np.concatenate((xs, ys), axis=2)

        if q2.shape[0] == 0:
            q_pixel_positions[2] = np.zeros((0, self.pixel_range.shape[0], 2))
        else:
            y_pixel_increments = y + self.pixel_range
            ys = np.tile(y_pixel_increments, (q2.shape[0], 1))
            slopes = np.reshape(np.tan(q2), (q2.shape[0], 1))
            bs = y - (slopes * x)

            xs = np.round((ys - bs) / slopes, 0).astype(int)
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            q_pixel_positions[2] = np.concatenate((xs, ys), axis=2)

        if q3.shape[0] == 0:
            q_pixel_positions[3] = np.zeros((0, self.pixel_range.shape[0], 2))
        else:
            x_pixel_increments = x - self.pixel_range
            xs = np.tile(x_pixel_increments, (q3.shape[0], 1))
            slopes = np.reshape(np.tan(q3), (q3.shape[0], 1))
            bs = y - (slopes * x)

            ys = np.round((xs * slopes) + bs, 0).astype(int)
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            q_pixel_positions[3] = np.concatenate((xs, ys), axis=2)

        if q4.shape[0] == 0:
            q_pixel_positions[4] = np.zeros((0, self.pixel_range.shape[0], 2))
        else:
            y_pixel_increments = y - self.pixel_range
            ys = np.tile(y_pixel_increments, (q4.shape[0], 1))
            slopes = np.reshape(np.tan(q4), (q4.shape[0], 1))
            bs = y - (slopes * x)

            xs = np.round((ys - bs) / slopes, 0).astype(int)
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            q_pixel_positions[4] = np.concatenate((xs, ys), axis=2)

        pixel_positions = np.concatenate(
            (q_pixel_positions[quadrants[0]], 
                q_pixel_positions[quadrants[1]], 
                q_pixel_positions[quadrants[2]], 
                q_pixel_positions[quadrants[3]]), axis=0)
        
        return pixel_positions

# Update weights based on sensor data
def updateWeights(ps_pos, ps_orientation, scan_data, pixel_matrix, map_metadata):
    laser_simulator = LaserSimulator(pixel_matrix, map_metadata)
    simulated_scans = laser_simulator.getParticleRanges(ps_pos, ps_orientation)
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

# Visualize particles, true position, and landmarks
def visualize_particles(ps_pos, true_pos, landmarks, step, lidar_data, pixel_matrix):
    plt.clf()
    plt.imshow(pixel_matrix, cmap='gray')
    plt.scatter(true_pos[0], true_pos[1], c='r', s=100, marker='x', label='True Position')
    plt.scatter(ps_pos[:, 0], ps_pos[:, 1], c='b', s=10, label='Particles')
    for landmark in landmarks:
        plt.scatter(landmark[0], landmark[1], c='g', s=100, marker='o', label='Landmark')
    plt.scatter(lidar_data[:, 0], lidar_data[:, 1], c='m', s=50, marker='.', label='LiDAR Data')
    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.legend()
    plt.title(f'Step {step}')
    plt.pause(0.01)

# Initialization
X_0 = initialize_state(3, 3, 0)
U = generate_control_sequence_square_pattern(40)
landmarks = [(1, 3), (4, 4)]
ps_pos = np.hstack((np.random.rand(100, 2) * 20, (np.random.rand(100, 1) * 2 * np.pi - np.pi)))  # Initialize particles randomly within a 20x20 area, including orientation
ps_orientation = ps_pos[:, 2].reshape(-1, 1)  # Extract the orientations
ps_weights = np.ones((100, 1)) / 100

# Create pixel matrix for the map
pixel_matrix = np.ones((20, 20))
pixel_matrix[5:15, 5:15] = 0

# Generate synthetic data
X_sequence, Z_sequence = generate_data(X_0, U, landmarks[0], 40)

# Set up the plot
plt.figure()

# Main loop for the micro-simulation
for t in range(40):
    v, omega = U[t]  # Use the control sequence values
    dist = v
    bearing = omega
    new_orientation = 0.01  # This can be adjusted as needed
    ps_pos, ps_orientation = predictionStep(ps_pos, ps_orientation, dist, bearing, new_orientation)
    
    # Update weights
    scan_data = Z_sequence[t]  # Using synthetic scan data
    ps_weights = updateWeights(ps_pos, ps_orientation, scan_data, pixel_matrix, {'resolution': 1})
    
    # Resampling
    ps_pos, ps_orientation, ps_weights = resampling(ps_pos, ps_orientation, ps_weights)
    
    # Simulate LiDAR data for visualization
    laser_simulator = LaserSimulator(pixel_matrix, {'resolution': 1})
    simulated_lidar = laser_simulator.getParticleRanges(np.array([[X_sequence[t][0], X_sequence[t][1], X_sequence[t][2]]]), np.array([X_sequence[t][2]]))
    lidar_points = np.array([[X_sequence[t][0] + r * np.cos(theta), X_sequence[t][1] + r * np.sin(theta)] 
                             for r, theta in zip(simulated_lidar[0], np.linspace(-np.pi, np.pi, len(simulated_lidar[0])))])
    
    # Visualize particles every 2 steps
    if t % 2 == 0:
        visualize_particles(ps_pos, X_sequence[t], landmarks, t, lidar_points, pixel_matrix)
        print(f'Step {t}, State: {ps_pos.mean(axis=0)}, Orientation: {ps_orientation.mean()}, Weight: {ps_weights.mean()}')

plt.show()
