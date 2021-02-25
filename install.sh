#!/bin/bash
sudo apt-get update -y
sudo apt-get upgrade -y

# install java jdk v1.8.0
sudo apt-get install openjdk-8-jdk

# check if 'java' is set to the correct version
# java check script retrieved from https://stackoverflow.com/questions/7334754/correct-way-to-check-java-version-from-bash-script
# with necessary alterations
# if type -p java; then
#     _java=java
# elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]]; then
#     _java="$JAVA_HOME/bin/java"
# else
#     echo "Java Not Found"
# fi

# if [[ "$_java" ]]; then
#     version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
#     if [[ "$version" > "1.8" ]]; then
#         sudo update-alternatives --remove-all java
#         sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java 1
#     else
#         echo "Java set to 1.8.0 already"
# fi


# install tools
sudo apt-get install net-tools
sudo apt install git wget build-essential　python3-dev python3-pip
sudo apt install libopencv-dev
sudo pip3 install opencv-python
sudo apt install cmake
sudo apt install make git g++  
sudo apt install pytorch torchvision opencv

#pull darknet and MiDas
git clone https://github.com/AlexeyAB/darknet.git
cd ./darknet
touch .gitignore
cd ../
git clone https://github.com/intel-isl/MiDaS.git
cd ./MiDaS
touch .gitignore
cd ../

# move darknet source edits into darknet
cp darknet_edits/demo.c ./darknet/src/
cp darknet_edits/image_opencv.cpp ./darknet/src
cp darknet_edits/image_opencv.h ./darknet/src
cp darknet_edits/Makefile ./darknet/

cd ./darknet
make
cd ../




# Apache Kafka Installation
cd ~/Downloads/
wget http://apache.claz.org/kafka/2.7.0/kafka_2.13-2.7.0.tgz

sudo mkdir /opt/kafka
sudo tar -xvf kafka_2.13-2.7.0.tgz -C /opt/kafka

rm -rf kafka_2.13-2.7.0.tgz
