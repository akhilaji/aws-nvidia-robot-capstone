from kafka import KafkaConsumer

obj_result_consumer = KafkaConsumer('objResult', bootstrap_servers=['localhost:9092'])
dep_result_consumer = KafkaConsumer('depResult', bootstrap_servers=['localhost:9092'])