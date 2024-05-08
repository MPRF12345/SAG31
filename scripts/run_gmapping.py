#!/usr/bin/env python

# Isto não chegou a funcionar

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from math import degrees
from gmapping import GMapping

class GMappingNode:
    def _init_(self):
        rospy.init_node('gmapping_node')
        rospy.on_shutdown(self.shutdown)

        # Subscribers
        rospy.Subscriber('odom', Odometry, self.odom_callback)
        rospy.Subscriber('scan', LaserScan, self.scan_callback)

        # GMapping
        self.gmapping = GMapping()
        self.gmapping.init()

    def odom_callback(self, msg):
        # Get robot's pose (position and orientation)
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Convert orientation to Euler angles (roll, pitch, yaw)
        (roll, pitch, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        # Print robot's pose
        rospy.loginfo("Robot Pose: x={}, y={}, yaw={}".format(position.x, position.y, degrees(yaw)))

        # Update GMapping with robot's pose
        self.gmapping.update_odom(position.x, position.y, yaw)

    def scan_callback(self, msg):
        # Update GMapping with laser scan data
        self.gmapping.update_scan(msg.ranges)

    def shutdown(self):
        rospy.loginfo("Shutting down GMapping node")
        # Save the map before shutting down (optional)
        self.gmapping.save_map("map")

if _name_ == '_main_':
    try:
        node = GMappingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass