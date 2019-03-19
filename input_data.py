# -*- coding:utf-8 -*-
import tensorflow as tf
import inference as ff

def read_and_decode_to_createBatchDatas(filename, batchsize):
	# 创建一个reader用来读取TFrecord中的记录
	reader = tf.TFRecordReader()
	# 创建一个队列来维护输入文件列表
	filename_queue = tf.train.string_input_producer([filename], shuffle=False)
	# 从文件中读取一个样列，也可以使用read_up_to来读取多个样例
	_, serialized_example = reader.read(filename_queue)
	print(_,serialized_example)

	# 解析读入的一个样例，如果需要解析多个，则使用parse_example
	features = tf.parse_single_example(
		serialized_example,
		features ={'image_raw':tf.FixedLenFeature([], tf.string),
					'lable':tf.FixedLenFeature([], tf.int64)})
	# 将字符串解析成图像对应的像素数组
	print(features)
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	# reshape成224*224*3通道图片
	image = tf.reshape(image,[ff.IMAGE_SIZE, ff.IMAGE_SIZE, ff.NUM_CHANNELS])
	images = tf.image.per_image_standardization(image)
	print('123')
	print(features['lable'])
	labels = tf.cast(features['lable'], tf.int32)
	print('124')

	# 生成批数据
	min_after_dequeue = 10
	capacity = min_after_dequeue + 3 * batchsize
	image_batch,label_batch = tf.train.shuffle_batch(
										[images, labels],
										batch_size = batchsize,
										capacity = capacity,
										min_after_dequeue=min_after_dequeue)
	label_batch = tf.one_hot(label_batch, depth=1)
	return image_batch,label_batch


