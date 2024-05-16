#!/usr/bin/env python

# Make a script that has global variable to store information from the topics it subscribes to. The topics are /tf and /scan

import rospy
import tf2_ros
import geometry_msgs.msg
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import numpy as np

class SubscriberNode:
    def __init__(self):
        rospy.init_node('subscriber_node')
        rospy.on_shutdown(self.shutdown)

        # Create a TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/tf', geometry_msgs.msg.TransformStamped, self.tf_callback)

        # Global variables
        self.scan_data = None
        self.oldest_tf_data = [0, 0, 0]
        self.first = True
        self.recent_tf_data = [0, 0, 0]

    def scan_callback(self, msg):
        self.scan_data = msg.ranges

    def tf_callback(self, msg):
        # Get the transform translation
        translation = msg.transform.translation
        rotation = msg.transform.rotation
        (roll, pitch, yaw) = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])

        if self.first:
            self.oldest_tf_data = [msg.transform.translation.x, msg.transform.translation.y, yaw]
            self.first = False

        self.recent_tf_data = [msg.transform.translation.x, msg.transform.translation.y, yaw]


    def getScanData(self):
        rospy.loginfo("Getting scan data ", self.scan_data)
        return self.scan_data

    def getOdomData(self):
        d = self.recent_tf_data - self.oldest_tf_data
        dist = np.sqrt(d[0]**2 + d[1]**2)
        bearing = np.arctan2(d[1], d[0]) - self.oldest_tf_data[2]
        new_orientation = self.recent_tf_data[2] - self.oldest_tf_data[2]
        return dist, bearing, new_orientation

    def shutdown(self):
        rospy.loginfo("Shutting down Subscriber Node")

if __name__ == '__main__':
    try:
        subscriber_node = SubscriberNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass