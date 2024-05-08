import numpy as np

def cell_to_world(map_resolution, map_origin, cell_x, cell_y, orientation):
    angle_to_origin = np.arctan2(-cell_y, cell_x)
    distance_to_origin = np.sqrt(cell_x**2 + cell_y**2)
    origin_x, origin_y, origin_angle = map_origin.position.x, map_origin.position.y, map_origin.orientation.z

    # Calculate real-world position
    real_x = origin_x + map_resolution * np.cos(origin_angle + angle_to_origin) * distance_to_origin
    real_y = origin_y + map_resolution * np.sin(origin_angle + angle_to_origin) * distance_to_origin
    
    # Calculate real-world orientation
    real_orientation = orientation + origin_angle
    
    return real_x, real_y, real_orientation


def intersecting_cell(map_width, map_height, cell_x, cell_y, orientation):
    
    # Calculate equation of the line passing through origin cell
    slope = np.tan(orientation)
    if np.isinf(slope):
        r = np.arange(cell_y, 0) if orientation > 0 else np.arange(cell_y, map_height)
        for y in r:
            if is_cell_occupied(cell_x, y):
                return cell_x, y

    intercept = -cell_y - slope * cell_x
    
    # Check which direction has the biggest incline
    if abs(slope) > 1:
        # Iterate over y-axis
        r = np.arange(cell_y, 0) if orientation > 0 else np.arange(cell_y, map_height)
        for y in r:
            x = round((-y - intercept) / slope, 0)
            if x < 0 or x >= map_width:
                return None, None
            if is_cell_occupied(x, y):
                return x, y
    else:
        # Iterate over x-axis
        r = np.arange(cell_x, 0) if abs(orientation) > (np.pi/2) else np.arange(cell_x, map_width)
        for x in r:
            y = -round(slope * x + intercept, 0)
            if y < 0 or y >= map_height:
                return None, None
            if is_cell_occupied(x, y):
                return x, y

    return None, None  # No intersection found

class Particle:
    def __init__(self, x, y, orientation, weight):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.weight = weight

    def update(self, x, y, orientation, weight):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.weight = weight

    def move(self, dx, dy, dorientation):
        self.x += dx
        self.y += dy
        self.orientation += dorientation

    def copy(self):
        return Particle(self.x, self.y, self.orientation, self.weight)
    

# Class for model of the laser scanner
class LaserScanner:
    def __init__(self, max_range, min_angle, max_angle, angle_increment):
        self.max_range = max_range
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_increment = angle_increment

    def simulate_scan(self, particle, map):
        angles = np.arange(self.min_angle, self.max_angle, self.angle_increment)
        ranges = []
        for angle in angles:
            x, y = intersecting_cell(map.width, map.height, particle.x, particle.y, particle.orientation + angle)
            if x is None:
                ranges.append(self.max_range)
            else:
                distance = np.sqrt((x - particle.x)**2 + (y - particle.y)**2)
                ranges.append(distance)
        return ranges



def is_cell_occupied(x, y):
    # Placeholder function, you need to implement your own logic to check if the cell is occupied
    # For example, querying the occupancy grid map
    return False
