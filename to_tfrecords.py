# -*- coding:utf-8 -*-
import os 
import tensorflow as tf 
from PIL import Image
'''
该脚本用于将训练数据准换为tfrecord 格式
'''
# 生成数值类型的属性
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成字符串类型的属性
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
data_path = 'C:\\Users\\LIULINYUAN\\Desktop\\birdge crack detect\\image'
map_file = 'C:\\Users\\LIULINYUAN\\Desktop\\birdge crack detect\\map_file.txt'
classes = {'crack'}
class_map={}
train_tfrecord='C:\\Users\LIULINYUAN\Desktop\\birdge crack detect\\train_tf.tfrecords'
test_tfrecord='C:\\Users\LIULINYUAN\Desktop\\birdge crack detect\\test_tf.tfrecords'
writer1 = tf.python_io.TFRecordWriter(train_tfrecord)
writer2 = tf.python_io.TFRecordWriter(test_tfrecord)
# file_tfrecord='file_tf.tfrecords'
# writer = tf.python_io.TFRecordWriter(file_tfrecord)
train_file = 'C:\\Users\LIULINYUAN\Desktop\\birdge crack detect\\train.txt'
test_file = 'C:\\Users\LIULINYUAN\Desktop\\birdge crack detect\\test.txt'
reader1 = open(train_file,'r+')
reader2 = open(test_file,'r+')
for index, name in enumerate(classes):
	print(index, name)
	class_map[index]=name
	class_path=data_path+'/'+name+'/'
	for image_name in reader1.readlines():
	# for image_name in os.listdir(class_path):
		print(image_name)
		image_name = image_name.replace("\n","")+'.jpg'
		image_path = class_path  +image_name #训练图片的路径
		img = Image.open(image_path)
		img = img.resize((224, 224)) #根据实际情况设置尺寸大小
		img_raw = img.tobytes() #将图片转换为二进制格式
		example=tf.train.Example(features=tf.train.Features(feature=
		{'lable': _int64_feature(index),
		 'image_raw': _bytes_feature(img_raw)
		}))
		writer1.write(example.SerializeToString()) #序列化字符串
	writer1.close()
	for image_name in reader2.readlines():
	# for image_name in os.listdir(class_path):
		print(image_name)
		image_name = image_name.replace("\n","")+'.jpg'
		image_path = class_path  +image_name #训练图片的路径
		img = Image.open(image_path)
		img = img.resize((224, 224)) #根据实际情况设置尺寸大小
		img_raw = img.tobytes() #将图片转换为二进制格式
		example=tf.train.Example(features=tf.train.Features(feature=
		{'lable': _int64_feature(index),
		 'image_raw': _bytes_feature(img_raw)
		}))
		writer2.write(example.SerializeToString()) #序列化字符串
	writer2.close()
txtfile = open(map_file, 'w')
for key in class_map.keys():
	txtfile.write(str(key)+':'+class_map[key]+'\n')
txtfile.close