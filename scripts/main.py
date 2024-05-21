import yaml
import numpy as np
from PIL import Image
from .subscribers import SubscriberNode
from .laser_simulator import LaserSimulator
import rospy
from scipy.special import softmax

def getMap():
    # get the .pgm and .yaml files on the maps/ folder
    map_file = 'maps/map1_cropped.pgm'
    map_metadata_file = 'maps/map1.yaml'

    image = Image.open(map_file)

    pixel_matrix = np.reshape(np.array(image.getdata()), image.size)

    map_metadata = None
    with open(map_metadata_file, 'r') as file:
        data = yaml.safe_load(file)
        map_metadata = {'resolution': data['resolution'], 'origin': data['origin']}

    return pixel_matrix, map_metadata
    

# main loop
if __name__ == '__main__':
    pixel_matrix, map_metadata = getMap()
    subscribers = SubscriberNode()
    laser_simulator = LaserSimulator(pixel_matrix, map_metadata) # create only inside updateWeights

    # TODO: Initialize particles
    ps_pos = np.zeros((100, 2))
    ps_orientation = np.zeros((100, 1))
    ps_weights = np.ones((100, 1))

    while not rospy.is_shutdown():
        dist, bearing, new_orientation = subscribers.getOdomData()
        scan_data = subscribers.getScanData()

        # prediction step
        ps_pos, ps_orientation = predictionStep(ps_pos, ps_orientation, dist, bearing, new_orientation)

        # update weights
        ps_weights = updateWeights(ps_pos, ps_orientation, scan_data, pixel_matrix, map_metadata)
        #errors_neg = -np.array(abs(errors))
        #weights = softmax(errors_neg)

        # resampling
        ps_pos, ps_orientation, ps_weights = resampling(ps_pos, ps_orientation, ps_weights)

        rospy.sleep(0.1)

    rospy.spin()



    
    
