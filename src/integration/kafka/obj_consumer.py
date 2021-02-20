from kafka import KafkaConsumer

obj_consumer = KafkaConsumer('vidInput',
                             group_id='camera-group',
                             bootstrap_servers=['localhost:9092'])

for msg in consumer:
    # message values are in raw bytes
    # e.g. for unicode: `msg.value.decode('utf-8')`

    print("%s:%d%d: key=%s value=%s" %(msg.topic, msg.partition,
                                       msg.offset, msg.key,
                                       msg.value.decode('utf-8')))
