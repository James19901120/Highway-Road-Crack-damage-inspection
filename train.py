# -*- coding:utf-8 -*-
import tensorflow as tf
import inference as ff
import input_data as id

from google.protobuf import text_format

from tensorflow.python.framework import graph_util

tfrecord_file = 'C:\\Users\\LIULINYUAN\\Desktop\\birdge crack detect\\train_tf.tfrecords'

save_models = 'C:\\Users\\LIULINYUAN\\Desktop\\birdge crack detect\\model'

Batch_size = 10
train_num = 6
learn_rate_base = 0.01
learn_rate_decay = 0.99
moving_averge_decay = 0.99
regularition_rate = 0.0001

x = tf.placeholder(tf.float32,[None, ff.IMAGE_SIZE, ff.IMAGE_SIZE, ff.NUM_CHANNELS], name = 'x_input')
y_ = tf.placeholder(tf.float32, [None, ff.NUM_LABELS], name = 'y_input')

image_batch, label_batch = id.read_and_decode_to_createBatchDatas(tfrecord_file, Batch_size)
print(image_batch, label_batch)

regularizer = tf.contrib.layers.l2_regularizer(regularition_rate)
print("x:", x)
y = ff.inference(x, True, regularizer)
print("y:", y)

global_step = tf.Variable(0, trainable=False)

variable_average = tf.train.ExponentialMovingAverage(moving_averge_decay, global_step)
variable_average_op = variable_average.apply(tf.trainable_variables())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_)

cross_entroy_mean = tf.reduce_mean(cross_entropy)

#loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
loss = cross_entroy_mean

learning_rate = tf.train.exponential_decay(learn_rate_base,
										  global_step,
										  train_num/Batch_size,
										  learn_rate_decay,
										  staircase = True
)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

prediction_labels=tf.argmax(y, axis=1,name = 'output')

correct_prediction= tf.equal(prediction_labels, tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.control_dependencies([train_step,variable_average_op]):
	train_op = tf.no_op('train')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
	tf.train.write_graph(sess.graph_def, save_models, 'graph.pbtxt', as_text=True)
	tf.train.write_graph(sess.graph_def, save_models, 'graph.pb', as_text=False)
	
	sess.run(init)
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	try:
		while not coord.should_stop():
			for i in range(train_num):
				print('start training')
				xs_batch, ys_batch =sess.run([image_batch, label_batch])
				print(xs_batch, ys_batch)
				_, train_accuracy = sess.run([train_op,accuracy], feed_dict={x:xs_batch, y_:ys_batch})
				print('Step %d, accuracy = %g' %(i,train_accuracy))
				if i % 2 == 0:
					saver.save(sess, save_models+'.ckpt', global_step = i)
			constant_graph = graph_util.convert_variables_to_constants(sess, 			sess.graph_def, ['output'])
			with tf.gfile.FastGFile(save_models+'model.pb',mode='wb' ) as f:
				f.write(constant_graph.SerializeToString())
	except tf.errors.OutOfRangeError:
		print('done training -- epoch limit reached')
	finally:
		coord.request_stop()

	coord.join(threads)				
			
			
