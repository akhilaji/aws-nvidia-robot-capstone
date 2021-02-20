from kafka import KafkaConsumer

obj_consumer = KafkaConsumer('vidInput',
                             group_id='camera-group',
                             bootstrap_servers=['localhost:9092'])
