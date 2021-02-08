#!/bin/bash
sudo apt-get update -y
sudo apt-get upgrade -y

# add and install java
sudo add-apt-repository -y ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer -y

# install zookeeper
sudo apt-get install zookeeperd

# download and install kafka
wget https://www-eu.apache.org/dist/kafka/2.1.1/kafka_2.11-2.1.1.tgz 
sudo mkdir /opt/Kafka
cd /opt/Kafka
sudo tar -xvf kafka_2.11-2.1.1.tgz -C /opt/Kafka/

# move darknet source edits into darknet
cp darknet_edit/demo.c ~/darknet/src/
cp darknet_edit/image_opencv.cpp ~/darknet/src
cp darknet_edit/image_opencv.h ~/darknet/src

