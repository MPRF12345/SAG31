# ist-meic-saut-g31

1. Clone repo into `~/catkin_ws/src`
2. `cd ~/catkin_ws`
3. `catkin_make --only-pkg-with-deps ist-meic-saut-g31`
4. `. ~/catkin_ws/devel/setup.bash`


`source /opt/ros/noetic/setup.bash`

Aceder a terminais no pioneer:
`ssh pi@192.168.28.20`
`acsdclsdc4`

Correr sempre: (no robo)
1. `roscore`
2. `rosrun p2os_driver p2os_driver _port:="/dev/ttyUSB0"`
3. (Para ligar o sensor/scan/lidar) `rosrun urg_node urg_node /dev/ttyACM0`


Antes de correr alguma coisa no computador:
1. Dar update aos exports no ficheiro `~/.bashrc` para o ip correto
2. Fechar e abrir o terminal ou correr `source ~/.bashrc` 

Pode-se correr no pioneer ou no computador:
1. (Para movimentar o robo) `rosrun teleop_twist_keyboard teleop_twist_keyboard.py`
