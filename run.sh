# sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix asu-aws-nvidia-robo
sudo docker run -i -t --net=host -e DISPLAY -v /tmp/.X11-unix asu-aws-nvidia-robo /bin/bash