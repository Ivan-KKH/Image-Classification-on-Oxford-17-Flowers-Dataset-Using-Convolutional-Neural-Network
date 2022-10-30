import tensorflow as tf


for event in tf.train.summary_iterator('runs/easy_name/events.out.tfevents.1521590363.DESKTOP-43A62TM'):
    for value in event.summary.value:
        print(value.tag)
        if value.HasField('simple_value'):
            print(value.simple_value)