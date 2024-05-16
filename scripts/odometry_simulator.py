#!/usr/bin/env python

# ISTO NAO É PARA SER USADO MAS PODE SER UTIL AINDA

import rospy
import tf2_ros
import geometry_msgs.msg
import numpy as np

def tf_broadcaster():
    # Initialize ROS node
    rospy.init_node('tf_simulator')

    # Create a TF broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Define the parent and child frames
    parent_frame = "parent_frame"
    child_frame = "child_frame"

    # Create a TransformStamped message
    transform_stamped = geometry_msgs.msg.TransformStamped()

    # Set the frame IDs
    transform_stamped.header.frame_id = parent_frame
    transform_stamped.child_frame_id = child_frame

    # Set the transform translation
    transform_stamped.transform.translation.x = 0.0
    transform_stamped.transform.translation.y = 0.0
    transform_stamped.transform.translation.z = 0.0

    # Set the transform rotation (no rotation in this example)
    transform_stamped.transform.rotation.x = 0.0
    transform_stamped.transform.rotation.y = 0.0
    transform_stamped.transform.rotation.z = 0.0
    transform_stamped.transform.rotation.w = 1.0

    # Publish the transform repeatedly
    rate = rospy.Rate(10)  # 1 Hz
    while not rospy.is_shutdown():
        transform_stamped.header.stamp = rospy.Time.now()
        # change the transforms randomly
        speed = np.random.uniform(0, 0.2)
        transform_stamped.transform.translation.x += speed * np.cos(transform_stamped.transform.rotation.z)
        transform_stamped.transform.translation.y += speed * np.sin(transform_stamped.transform.rotation.z)
        transform_stamped.transform.rotation.z += np.random.uniform(-0.1, 0.)

        tf_broadcaster.sendTransform(transform_stamped)
        rate.sleep()

if __name__ == '__main__':
    try:
        tf_broadcaster()
    except rospy.ROSInterruptException:
        pass
