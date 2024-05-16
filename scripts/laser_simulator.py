import numpy as np

class LaserSimulator:
    def __init__(self, pixel_matrix, map_metadata):
        self.laser_scanner = {
            'min_range': 0.02, # meters
            'max_range': 5.6, # meters
            'min_angle': -120 * np.pi / 180,
            'max_angle': 120 * np.pi / 180,
            'angle_increment': 0.352 * np.pi / 180
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

        for i in range(n_particles):
            x, y = ps_pos[i]
            orientation = ps_orientation[i]

            orientations = orientation + self.scan_angles
            # to keep all agles between -pi and pi
            orientations = np.mod(orientations + np.pi, 2 * np.pi) - np.pi

            pixel_positions = self._getPixelPositions(orientations, x, y)

            # get the values of the pixels
            pixel_values = self.pixel_matrix[pixel_positions[:, :, 0], pixel_positions[:, :, 1]]

            # find the first occupied pixel in each row
            occupied_pixels = np.argmax(pixel_values, axis=1)

            # get the position of the occupied pixels
            occupied_positions = pixel_positions[np.arange(pixel_positions.shape[0]), occupied_pixels]

            # get the distance to the occupied pixels
            distances = np.sqrt(np.sum((occupied_positions - np.array([[x, y]])) ** 2, axis=1))

            # get the ranges
            ranges = distances * self.map_metadata['resolution']

            return ranges

    def _getPixelPositions(self, orientations, x, y):

        # division of orientations into the 4 quadrants cut by 45 degrees
        q1 = orientations[orientations > -np.pi / 4 and orientations < np.pi / 4]
        q2 = orientations[orientations > np.pi / 4 and orientations < 3 * np.pi / 4]
        q3 = orientations[orientations > 3 * np.pi / 4 or orientations < -3 * np.pi / 4]
        q4 = orientations[orientations > -3 * np.pi / 4 and orientations < -np.pi / 4]

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

        q_pixel_positions = {}
        # q1
        if q1.shape[0] == 0:
            q_pixel_positions[1] = np.zeros((0, self.pixel_range.shape[0] , 2))
        x_pixel_increments = x + self.pixel_range
        xs = np.tile(x_pixel_increments, (q1.shape[0], 1))
        slopes = np.reshape(np.tan(q1), (q1.shape[0], 1))
        bs = y - (slopes * x)

        ys = np.round((xs * slopes) + bs, 0)
        ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
        xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
        q_pixel_positions[1] = np.concatenate((xs, ys), axis=2)

        # q2
        if q2.shape[0] == 0:
            q_pixel_positions[2] = np.zeros((0, self.pixel_range.shape[0] , 2))
        y_pixel_increments = y + self.pixel_range
        ys = np.tile(y_pixel_increments, (q2.shape[0], 1))
        slopes = np.reshape(np.tan(q2), (q2.shape[0], 1))
        bs = y - (slopes * x)

        xs = np.round((ys - bs) / slopes, 0)
        xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
        ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
        q_pixel_positions[2] = np.concatenate((xs, ys), axis=2)

        # q3
        if q3.shape[0] == 0:
            q_pixel_positions[3] = np.zeros((0, self.pixel_range.shape[0] , 2))
        x_pixel_increments = x - self.pixel_range
        xs = np.tile(x_pixel_increments, (q3.shape[0], 1))
        slopes = np.reshape(np.tan(q3), (q3.shape[0], 1))
        bs = y - (slopes * x)

        ys = np.round((xs * slopes) + bs, 0)
        ys = np.reshape(ys, (ys.shape[0], ys.shape[1], 1))
        xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
        q_pixel_positions[3] = np.concatenate((xs, ys), axis=2)

        
        # q4
        if q4.shape[0] == 0:
            q_pixel_positions[4] = np.zeros((0, self.pixel_range.shape[0] , 2))
        y_pixel_increments = y + self.pixel_range
        ys = np.tile(y_pixel_increments, (q4.shape[0], 1))
        slopes = np.reshape(np.tan(q4), (q4.shape[0], 1))
        bs = y - (slopes * x)

        xs = np.round((ys - bs) / slopes, 0)
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
                
