from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")
import argparse
import sys
import numpy as np
from dataset import CifarData
from mobilenet import MobileNet
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.chdir(os.getcwd())

tf.reset_default_graph()

slim = tf.contrib.slim


def main(args):
	"""
	This is the main function which runs the algorithm
	:param args: The arguments for the algorithm
	:return: NA
	"""
	print_arguments_given(args)
	input = tf.placeholder(dtype = tf.float32, shape = (None, 3072), name = 'input_data')
	actual_y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'input_label')
	is_train = tf.placeholder(dtype = tf.bool, name = 'is_train')
	global_step = tf.Variable(0, trainable = False)

	reshape_x = tf.reshape(input, [-1, 3, 32, 32])

	reshape_x = tf.transpose(reshape_x, perm = [0, 2, 3, 1])

	with tf.variable_scope("MobileNet"):
		output = MobileNet(reshape_x, is_train).outputs
		avg_pooling = tf.nn.avg_pool(output, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = "SAME",
		                             name = "Avg_pooling")

		dense1 = tf.layers.dense(inputs = avg_pooling, units = 512, activation = None,
		                         kernel_initializer = tf.random_normal_initializer(stddev = 0.01), trainable = True,
		                         name = "dense1")

		bn1 = tf.layers.batch_normalization(dense1, beta_initializer = tf.zeros_initializer(),
		                                    gamma_initializer = tf.ones_initializer(),
		                                    moving_mean_initializer = tf.zeros_initializer(),
		                                    moving_variance_initializer = tf.ones_initializer(), training = is_train,
		                                    name = 'bn1')
		relu1 = tf.nn.leaky_relu(bn1, 0.1)

		dense2 = tf.layers.dense(inputs = relu1, units = 10,
		                         kernel_initializer = tf.random_normal_initializer(stddev = 0.01), trainable = True,
		                         name = "dense2")
		sqz = tf.squeeze(dense2, [1, 2], name = 'sqz')

		prediction = tf.nn.softmax(sqz, name = 'prediction')

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = actual_y, logits = sqz))
	predict = tf.argmax(prediction, 1)
	correct_prediction = tf.equal(predict, actual_y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
	moving_average = tf.train.ExponentialMovingAverage(0.99).apply(tf.trainable_variables())
	train_filename = [os.path.join('./cifar-10-batches-py', 'data_batch_%d' % i) for i in range(1, 6)]
	test_filename = [os.path.join('./cifar-10-batches-py', 'test_batch')]
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		with tf.control_dependencies([moving_average]):
			train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('We are going to start the training...')
		for epoch in range(1, args.epochs + 1):
			print("The current epoch is", epoch)
			train_data = CifarData(train_filename)
			for i in range(10000):
				batch_data, batch_labels, = train_data.next_batch(args.bs)
				loss_val, acc_val, _, predict1, yt = sess.run([loss, accuracy, train_op, predict, actual_y],
				                                              feed_dict = {input   : batch_data, actual_y:
					                                              batch_labels,
				                                                           is_train: True})
				if i == 9999:
					test_data = CifarData(test_filename)
					test_acc_sum = []
					for j in range(100):
						test_batch_data, test_batch_labels = test_data.next_batch(args.bs)
						test_acc_val = sess.run([accuracy],
						                        feed_dict = {input   : test_batch_data, actual_y: test_batch_labels,
						                                     is_train: False})
						test_acc_sum.append(test_acc_val)
					test_accuracy = np.mean(test_acc_sum)
					print('[Test ] accuracy: %4.5f' % test_accuracy)


def print_arguments_given(args):
	"""
	This function is used to print the arguments given in cmd
	:param args: The arguments
	:return: N/A
	"""
	print('=' * 1000)
	print('learning rate    : {}'.format(args.lr))
	print('Batch size       : {}'.format(args.bs))
	print('Epoch            : {}'.format(args.epochs))
	print('=' * 100)


def parse(argv):
	"""
	This function is used to parse the arguments
	:param argv: The arguments
	:return: THe parsed dictionary
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr',
	                    type = float,
	                    help = 'set learning rate',
	                    default = 1e-2)

	parser.add_argument('--bs',
	                    type = int,
	                    nargs = '+',
	                    help = 'Batch Size to train',
	                    default = 64)

	parser.add_argument('--epochs',
	                    type = int,
	                    nargs = '+',
	                    default = 10,
	                    help = 'Train Epochs')

	return parser.parse_args(argv)


if __name__ == '__main__':
	main(parse(sys.argv[1:]))
