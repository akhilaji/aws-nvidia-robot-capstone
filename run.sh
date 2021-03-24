#!/bin/bash

#start kafka
#run kafka producer
#run kafka consumers


sudo /opt/Kafka/kafka_2.13-2.7.0/bin/kafka-server-start.sh /opt/Kafka/kafka_2.13-2.7.0/config/server.properties &
sudo /opt/kafka_2.13-2.7.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1  --partitions 2 --topic vidInput &
sudo /opt/kafka_2.13-2.7.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1  --partitions 2 --topic objResult &
sudo /opt/kafka_2.13-2.7.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1  --partitions 2 --topic depResult &

#sleep 10s
#python3 ./src/integration/kafka/cam_producer.py &
#sleep 5s
#python3 ./src/integration/kafka/dep_consumer.py &
#sleep 5s
#python3 ./src/integration/kafka/obj_consumer.py &