# -*- coding:utf-8 -*-
import tensorflow as tf
'''
网络模型
'''
IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_LABELS = 1

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512 

def inference(input_tensor, train, regularizer):
	with tf.variable_scope('layer1_conv1'):
		conv1_weight = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
						initializer = tf.truncated_normal_initializer(stddev=0.1))
		conv1_bias = tf.get_variable('bias',[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1, 1, 1, 1], padding = 'SAME')
		conv1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
		
		
	with tf.name_scope('layer2_pool1'):
		pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1],strides=[1, 2, 2, 1], padding = 'SAME')
	with tf.variable_scope('layer3_conv2'):
		conv2_weight = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
						initializer = tf.truncated_normal_initializer(stddev=0.1))
		conv2_bias = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv2 = tf.nn.conv2d(pool1, conv2_weight, strides = [1, 1 ,1, 1],padding = 'SAME')
		conv2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
		
	with tf.name_scope('layer4_pool2'):
		pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	shp = pool2.get_shape()
	flattend_shape = shp[1].value * shp[2].value * shp[3].value
	reshape = tf.reshape(pool2,[-1, flattend_shape])
	
	with tf.variable_scope('layer5_fc1'):
		fc1_weight = tf.get_variable('weight', [flattend_shape, FC_SIZE],
							initializer = tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weight))
		fc1_bias = tf.get_variable('bias', [FC_SIZE], initializer= tf.constant_initializer(0.1))
		
		fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weight) + fc1_bias )
		
		if train:
			fc1 = tf.nn.dropout(fc1, 0.5)
	with tf.variable_scope('layer6_fc2'):
		fc2_weight = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
						initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None : tf.add_to_collection('losses', regularizer(fc2_weight))
		fc2_bias = tf.get_variable('bias', [NUM_LABELS],initializer = tf.constant_initializer(0.1))
		#shold not use relu for the output layer
		logit = tf.matmul(fc1, fc2_weight) + fc2_bias
	
	fileoutput = tf.nn.softmax(logit, name = 'softmax')
	return fileoutput
		
						
		
	
		
