Com uma bag:
- roscore
- rosparam set use_sim_time true
- rosrun tf static_transform_publisher 0 0 0 0 0 0 1 /base_link laser 100
- rosrun rviz rviz -d catkin_ws/src/ist-meic-saut-g31/launch/gmapping.rviz (para ver o mapa a ser criado)
- rosbag play 2024-04-30-20-46-52.bag --clock
- rosrun gmapping slam_gmapping
- rosrun map_server map_saver

Save also results from this topic: `map_metadata` (??)

OU

```roslaunch ist-meic-saut-g31 gmapping.launch```

Coisas a falar sobre o gmapping:
Ao mudar os parametros do static_transform_publisher (x de 1 para 0) o mapa perdeu o erro, mas está desalinhado, o que torna a ligação com o mundo real um pouco mais complicada (e fica feio). Isto pode ter ocorrido porque a posição inicial do robo (base_link, ou topico /pose) é (5, -5), provavelmente porque fizemos muitas experiencias antes que nao sabiamos que podiam influenciar. [Adicionar imagem do mapa antigo, em que os eixos mostram o 0,0 do mapa e a ultima posição do base_link]
Outra coisa que notamos foi a falta dos pontos das cadeiras e algumas medidas que parecia que o gmapping estava a receber muito maiores do que a sala. Como não há janelas, assumimos que esses pontos vem de reflexões nas cadeiras (porque são de metal), que também não aparecem no mapa.
Por isso, fizemos mais uma bag, sem mover previamente o robo, para alinhar os resultados, e editamos a imagem no paint para remover as reflexões das cadeiras.


(Isto é para outra coisa ignorem)
void mapToWorld(unsigned int map_x, unsigned int map_y, double& pos_x, double& pos_y, nav_msgs::OccupancyGrid map)
{
    pos_x = map.info.origin.position.x + (map_x + 0.5) * map.info.resolution;
    pos_y = map.info.origin.position.y + (map_y + 0.5) * map.info.resolution;
}
