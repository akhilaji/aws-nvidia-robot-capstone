#!/bin/bash
sudo apt-get update -y
sudo apt-get upgrade -y

# install java jdk v1.8.0
sudo apt-get install openjdk-8-jdk

# check if 'java' is set to the correct version
# java check script retrieved from https://stackoverflow.com/questions/7334754/correct-way-to-check-java-version-from-bash-script
# with necessary alterations

if type -p java; then
    _java=java
elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]]; then
    _java="$JAVA_HOME/bin/java"
else
    echo "Java Not Found"
fi

if [[ "$_java" ]]; then
    version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    if [[ "$version" > "1.8" ]]; then
        sudo update-alternatives --remove-all java
        sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java 1
    else
        echo "Java set to 1.8.0 already"
fi

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

