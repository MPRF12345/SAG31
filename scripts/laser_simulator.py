import numpy as np

colors = ['b', 'g', 'm', 'c', 'y', 'r']


class LaserSimulator:
    def __init__(self, pixel_matrix, map_metadata):
        self.laser_scanner = {
            'min_range': 0.02, # meters
            'max_range': 5.6, # meters
            'min_angle': -120 * np.pi / 180,
            'max_angle': 120 * np.pi / 180,
            'angle_increment': 3.352 * np.pi / 180
            # 'angle_increment': np.pi / 3
        }
        self.pixel_matrix = pixel_matrix
        self.map_metadata = map_metadata

        # get max number of pixels for range
        max_range_pixels = int(self.laser_scanner['max_range'] / self.map_metadata['resolution'])
        self.pixel_range = np.arange(1, max_range_pixels + 1)
        self.n_scans = int((self.laser_scanner['max_angle'] - self.laser_scanner['min_angle']) / self.laser_scanner['angle_increment'])
        self.scan_angles = np.linspace(self.laser_scanner['min_angle'], self.laser_scanner['max_angle'], self.n_scans)

    def getParticleRanges(self, ps_pos, ps_orientation, ax):
        n_particles = ps_pos.shape[0]

        ranges = np.zeros((n_particles, self.n_scans))


        print(n_particles)

        for i in range(n_particles):
            x, y = ps_pos[i]
            orientation = ps_orientation[i]

            print(self.scan_angles)

            orientations = orientation + self.scan_angles
            print(orientations)
            # to keep all agles between -pi and pi
            orientations = np.mod(orientations + np.pi, 2 * np.pi) - np.pi
            print(orientations)

            pixel_positions = self._getPixelPositions(orientations, x, y).astype(int)

            height, width = self.pixel_matrix.shape
            pixel_positions[:,:,0] = np.clip(pixel_positions[:,:,0], 0, width - 1)
            pixel_positions[:,:,1] = np.clip(-pixel_positions[:,:,1], 0, height - 1)

            ax.scatter(pixel_positions[:, :, 0], pixel_positions[:, :, 1], c='r', marker='.', label='Laser')

            # plt.show()

            print(pixel_positions[0])
            # get the values of the pixels
            pixel_values = self.pixel_matrix[pixel_positions[:, :, 1], pixel_positions[:, :, 0]]

            # find the first occupied pixel in each row
            occupied_pixels = np.argmax(pixel_values, axis=1)


            # get the position of the occupied pixels
            occupied_positions = pixel_positions[np.arange(pixel_positions.shape[0]), occupied_pixels]

            print("ocupos",occupied_positions)


            ax.scatter(occupied_positions[:, 0], occupied_positions[:, 1], c='b', marker='.', label='Particles')


            # get the distance to the occupied pixels
            distances = np.sqrt(np.sum((occupied_positions - np.array([[x, y]])) ** 2, axis=1))

            # get the ranges
            ranges[i] = distances * self.map_metadata['resolution']

            plt.show()
            
            return ranges

    def _getPixelPositions(self, orientations, x, y):

        y = -y

        # division of orientations into the 4 quadrants cut by 45 degrees
        q1 = orientations[(orientations > -np.pi / 4) & (orientations < np.pi / 4)]
        q2 = orientations[(orientations > np.pi / 4) & (orientations < 3 * np.pi / 4)]
        q3 = orientations[(orientations > 3 * np.pi / 4) | (orientations < -3 * np.pi / 4)]
        q4 = orientations[(orientations > -3 * np.pi / 4) & (orientations < -np.pi / 4)]

        # get order of quadrants
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

        print(quadrants)
        print(q1, q2, q3, q4)

        q_pixel_positions = {}
        # q1
        if q1.shape[0] == 0:
            q_pixel_positions[1] = np.zeros((0, self.pixel_range.shape[0] , 2))
        else:
            x_pixel_increments = x + self.pixel_range
            xs = np.tile(x_pixel_increments, (q1.shape[0], 1))
            slopes = np.reshape(np.tan(q1), (q1.shape[0], 1))
            bs = y - (slopes * x)

            ys = np.round((xs * slopes) + bs, 0).astype(int)
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            q_pixel_positions[1] = np.concatenate((xs, ys), axis=2)


        # q2
        if q2.shape[0] == 0:
            q_pixel_positions[2] = np.zeros((0, self.pixel_range.shape[0] , 2))
        else:
            y_pixel_increments = y + self.pixel_range
            ys = np.tile(y_pixel_increments, (q2.shape[0], 1))
            slopes = np.reshape(np.tan(q2), (q2.shape[0], 1))
            bs = y - (slopes * x)

            xs = np.round((ys - bs) / slopes, 0).astype(int)
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            q_pixel_positions[2] = np.concatenate((xs, ys), axis=2)

        # q3
        if q3.shape[0] == 0:
            q_pixel_positions[3] = np.zeros((0, self.pixel_range.shape[0] , 2))
        else:
            x_pixel_increments = x - self.pixel_range
            xs = np.tile(x_pixel_increments, (q3.shape[0], 1))
            slopes = np.reshape(np.tan(q3), (q3.shape[0], 1))
            bs = y - (slopes * x)

            ys = np.round((xs * slopes) + bs, 0).astype(int)
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            q_pixel_positions[3] = np.concatenate((xs, ys), axis=2)

            
        # q4
        if q4.shape[0] == 0:
            q_pixel_positions[4] = np.zeros((0, self.pixel_range.shape[0] , 2))
        else:
            y_pixel_increments = y - self.pixel_range
            ys = np.tile(y_pixel_increments, (q4.shape[0], 1))
            slopes = np.reshape(np.tan(q4), (q4.shape[0], 1))
            bs = y - (slopes * x)

            xs = np.round((ys - bs) / slopes, 0).astype(int)
            xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
            ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
            q_pixel_positions[4] = np.concatenate((xs, ys), axis=2)

        # concatenate all quadrants according to the order
        pixel_positions = np.concatenate(
            (q_pixel_positions[quadrants[0]], 
                q_pixel_positions[quadrants[1]], 
                q_pixel_positions[quadrants[2]], 
                q_pixel_positions[quadrants[3]]), axis=0)
        
        return pixel_positions
                

# code for testing the class

import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def getMap():
    # get the .pgm and .yaml files on the maps/ folder
    map_file = 'maps/map1_cropped.pgm'
    map_metadata_file = 'maps/map1.yaml'

    image = Image.open(map_file)

    image_pixel_matrix = np.reshape(np.array(image.getdata()), (image.size[1],image.size[0]))
    pixel_matrix = np.zeros_like(image_pixel_matrix)

    pixel_matrix[image_pixel_matrix == 0] = 2
    pixel_matrix[(image_pixel_matrix < 254) & (image_pixel_matrix > 0)] = 1
    pixel_matrix[image_pixel_matrix == 254] = 0

    map_metadata = None
    with open(map_metadata_file, 'r') as file:
        data = yaml.safe_load(file)
        map_metadata = {'resolution': data['resolution'], 'origin': data['origin']}

    return pixel_matrix, map_metadata, image_pixel_matrix
    
def cell_to_world(map_resolution, map_origin, cell_positions, cell_orientations):
    origin_x, origin_y, origin_angle = map_origin.position.x, map_origin.position.y, map_origin.orientation.z
    
    cell_x, cell_y = cell_positions[:, 0], cell_positions[:, 1]
    
    angle_to_origin = np.arctan2(-cell_y, cell_x)
    distance_to_origin = np.sqrt(cell_x**2 + cell_y**2)
    
    real_x = origin_x + map_resolution * np.cos(origin_angle + angle_to_origin) * distance_to_origin
    real_y = origin_y + map_resolution * np.sin(origin_angle + angle_to_origin) * distance_to_origin
    
    real_orientations = cell_orientations + origin_angle
    
    return np.column_stack((real_x, real_y)), real_orientations

def world_to_cell(map_resolution, map_origin, real_positions, real_orientations):
    origin_x, origin_y, origin_angle = map_origin
    
    real_x, real_y = real_positions[:, 0], real_positions[:, 1]
    
    relative_x = (real_x - origin_x) / map_resolution
    relative_y = (real_y - origin_y) / map_resolution
    
    distance_to_origin = np.sqrt(relative_x**2 + relative_y**2)
    angle_to_origin = np.arctan2(relative_y, relative_x)
    
    cell_x = distance_to_origin * np.cos(angle_to_origin - origin_angle)
    cell_y = -distance_to_origin * np.sin(angle_to_origin - origin_angle)
    
    cell_orientations = real_orientations - origin_angle
    
    return np.column_stack((cell_x.astype(int), cell_y.astype(int))), cell_orientations.astype(int)

# main loop
if __name__ == '__main__':
    pixel_matrix, map_metadata, image_pixel_matrix = getMap()
    # subscribers = SubscriberNode()
    laser_simulator = LaserSimulator(pixel_matrix, map_metadata) # create only inside updateWeights

    # TODO: Initialize particles
    ps_pos = np.array([[0, 0], [1, -3], [3, -5], [2, -4], [4, 0], [3, -2]])
    ps_orientation = np.reshape(np.arange(0, 2 * np.pi, 2 * np.pi / 6), (6, 1))
    ps_weights = np.ones((6, 1))


    # Plot the map, with the positions and orientation of the particles

    fig, ax = plt.subplots()
    ax.imshow(image_pixel_matrix, cmap='gray')

    ps_pixel_pos, ps_pixel_pos_o = world_to_cell(map_metadata['resolution'],(-5,3,0), ps_pos, ps_orientation)
    print(ps_pixel_pos, ps_pixel_pos_o)
    
    # Plot particle positions
    ax.scatter(ps_pixel_pos[:, 0], ps_pixel_pos[:, 1], c=colors, marker='o', label='Particles')

    # Plot particle orientations
    for i in range(len(ps_pixel_pos)):
        x, y = ps_pixel_pos[i]
        dx = np.cos(ps_pixel_pos_o[i])[0] * 6
        dy = np.sin(ps_pixel_pos_o[i])[0] * 6
        ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='b', ec='b')

    
    ranges = laser_simulator.getParticleRanges(ps_pixel_pos, ps_pixel_pos_o, ax)

    print(ranges)

    plt.show()



    
    