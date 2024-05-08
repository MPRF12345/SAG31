Com uma bag:
- roscore
- rosparam set use_sim_time true
- rosrun tf static_transform_publisher 0 0 0 0 0 0 1 /base_link laser 100
- rosrun rviz rviz -d catkin_ws/src/ist-meic-saut-g31/launch/gmapping.rviz (para ver o mapa a ser criado)
- rosbag play 2024-04-30-20-46-52.bag --clock
- rosrun gmapping slam_gmapping
- rosrun map_server map_saver

Save also results from this topic: `map_metadata` (??)


(Isto é para outra coisa ignorem)
void mapToWorld(unsigned int map_x, unsigned int map_y, double& pos_x, double& pos_y, nav_msgs::OccupancyGrid map)
{
    pos_x = map.info.origin.position.x + (map_x + 0.5) * map.info.resolution;
    pos_y = map.info.origin.position.y + (map_y + 0.5) * map.info.resolution;
}
