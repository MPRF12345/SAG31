import numpy as np

def sample_motion_model(n, alpha, yaw, d, new_alpha, x_variables, y_variables):
    """
    Update the particles' positions and angles based on odometry data.

    :param n: Number of particles.
    :param alpha: Current yaw angles of the particles.
    :param yaw: Yaw angles from odometry.
    :param d: Distance moved by the robot from odometry.
    :param new_alpha: New yaw angles from odometry.
    :param x_variables: Current x positions of the particles.
    :param y_variables: Current y positions of the particles.
    :return: Updated x positions, y positions, and yaw angles of the particles.
    """
    
    # Step 1: Add yaw angles to the current yaw angles (alpha)
    updated_yaw = alpha + yaw

    # Step 2: Calculate new x and y positions
    cos_result = np.cos(updated_yaw)
    sin_result = np.sin(updated_yaw)
    delta_x = d * cos_result
    delta_y = d * sin_result
    new_x_positions = x_variables + delta_x
    new_y_positions = y_variables + delta_y

    # Step 3: Update angles with new_alpha
    updated_alpha = alpha + new_alpha

    return new_x_positions, new_y_positions, updated_alpha

# Example usage
# n_particles = 100
# alpha = np.random.uniform(-np.pi, np.pi, n_particles)  # Example current yaw angles
# yaw = np.random.uniform(-0.1, 0.1, n_particles)  # Example yaw changes from odometry
# d = np.random.uniform(0, 1, n_particles)  # Example distances from odometry
# new_alpha = np.random.uniform(-0.1, 0.1, n_particles)  # Example new yaw angles from odometry
# x_variables = np.random.uniform(0, 10, n_particles)  # Example current x positions
# y_variables = np.random.uniform(0, 10, n_particles)  # Example current y positions

# new_x, new_y, updated_alpha = sample_motion_model(n_particles, alpha, yaw, d, new_alpha, x_variables, y_variables)

# print("New x positions:", new_x)
# print("New y positions:", new_y)
# print("Updated yaw angles:", updated_alpha)
