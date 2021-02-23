#!/bin/bash

#start kafka
#run kafka producer
#run kafka consumers


sudo /opt/Kafka/kafka_2.13-2.7.0/bin/kafka-server-start.sh /opt/Kafka/kafka_2.13-2.7.0/config/server.properties &
sudo /opt/kafka_2.13-2.7.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1  --partitions 1 --topic vidInput&

sleep 60s
python3 ./Kafka/producer.py &
sleep 60s
python3 ./Kafka/consumer.py &