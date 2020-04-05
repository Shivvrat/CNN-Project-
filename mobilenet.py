from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MobileNet:
	"""
	This is the MobileNet main class which will be used to do the image classification
	"""

	def __init__(self, input, trainable):
		self.input = input
		self.trainable = trainable
		self.outputs = self.__build_network__()

	def separable_conv_block(self, input, dw_filter, output_channel, strides, name):
		"""
		This method defines the structure for one downsample of the input
		:param input: The input to the unit
		:param dw_filter: The size of the filter
		:param output_channel: The number of channels in output
		:param strides: The stride of the filter
		:param name: The name of the layer
		:return: The output after the layer
		"""
		with tf.variable_scope(name):
			# We firstly find the weights for the layer by random initialization
			depthwise_weight = tf.get_variable(name = 'dw_filter', dtype = tf.float32, trainable = True,
			                                   shape = dw_filter,
			                                   initializer = tf.random_normal_initializer(stddev = 0.01))
			# Here we find the output of the depthwise convolution
			depthwise_output = tf.nn.depthwise_conv2d(input = input, filter = depthwise_weight, strides = strides,
			                                          padding = "SAME", name = 'Conv/depthwise_output')
			# Now we do the batch_wise normalization of the output to reduce effect of exploding and diminishing
			# gradient
			batch_normalization_depthwise = tf.layers.batch_normalization(depthwise_output,
			                                                              beta_initializer = tf.zeros_initializer(),
			                                                              gamma_initializer = tf.ones_initializer(),
			                                                              moving_mean_initializer =
			                                                              tf.zeros_initializer(),
			                                                              moving_variance_initializer =
			                                                              tf.ones_initializer(),
			                                                              training = self.trainable,
			                                                              name = 'depthwise_output/bn')
			# Now we take the rectified Linear Unit activation of the output
			relu = tf.nn.leaky_relu(batch_normalization_depthwise, 0.1)
			# Now we get the weight for the 2d convolution
			weight = tf.get_variable(name = 'weight', dtype = tf.float32, trainable = True,
			                         shape = (1, 1, dw_filter[2] * dw_filter[3], output_channel),
			                         initializer = tf.random_normal_initializer(stddev = 0.01))

			conv = tf.nn.conv2d(input = relu, filter = weight, strides = [1, 1, 1, 1], padding = "SAME",
			                    name = "conv/s1")
			batch_normalization_2d_conv = tf.layers.batch_normalization(conv, beta_initializer =
			tf.zeros_initializer(),
			                                                            gamma_initializer = tf.ones_initializer(),
			                                                            moving_mean_initializer =
			                                                            tf.zeros_initializer(),
			                                                            moving_variance_initializer =
			                                                            tf.ones_initializer(),
			                                                            training = self.trainable,
			                                                            name = 'pt/bn')
			return tf.nn.leaky_relu(batch_normalization_2d_conv, 0.1)

	def __build_network__(self):
		"""
		This function helps to build the network
		:return: The values of the output
		"""
		with tf.variable_scope('MobileNet'):
			convolution_1 = tf.layers.conv2d(self.input,
			                                 filters = 32,
			                                 kernel_size = (3, 3),
			                                 strides = (2, 2),
			                                 padding = 'same',
			                                 activation = tf.nn.relu,
			                                 name = 'convolution_1'
			                                 )
			batch_normalized_output_1 = tf.layers.batch_normalization(convolution_1,
			                                                          beta_initializer = tf.zeros_initializer(),
			                                                          gamma_initializer = tf.ones_initializer(),
			                                                          moving_mean_initializer = tf.zeros_initializer(),
			                                                          moving_variance_initializer =
			                                                          tf.ones_initializer(),
			                                                          training = self.trainable,
			                                                          name = 'bn')
			x = self.separable_conv_block(input = batch_normalized_output_1, dw_filter = (3, 3, 32, 1),
			                              output_channel = 64,
			                              strides = (1, 1, 1, 1), name = "downsample_1")

			x = self.separable_conv_block(input = x, dw_filter = (3, 3, 64, 1), output_channel = 128,
			                              strides = (1, 2, 2, 1), name = "downsample_2")

			x = self.separable_conv_block(input = x, dw_filter = (3, 3, 128, 1), output_channel = 128,
			                              strides = (1, 1, 1, 1), name = "downsample_3")

			x = self.separable_conv_block(input = x, dw_filter = (3, 3, 128, 1), output_channel = 256,
			                              strides = (1, 2, 2, 1), name = "downsample_4")

			x = self.separable_conv_block(input = x, dw_filter = (3, 3, 256, 1), output_channel = 256,
			                              strides = (1, 1, 1, 1), name = "downsample_5")
			x = self.separable_conv_block(input = x, dw_filter = (3, 3, 256, 1), output_channel = 512,
			                              strides = (1, 2, 2, 1), name = "downsample_6")

			for i in range(5):
				x = self.separable_conv_block(input = x, dw_filter = (3, 3, 512, 1), output_channel = 512,
				                              strides = (1, 1, 1, 1), name = "downsample_%d" % (i + 7))
			x = self.separable_conv_block(input = x, dw_filter = (3, 3, 512, 1), output_channel = 1024,
			                              strides = (1, 2, 2, 1), name = "downsample_12")

			x = self.separable_conv_block(input = x, dw_filter = (3, 3, 1024, 1), output_channel = 1024,
			                              strides = (1, 1, 1, 1), name = "downsample_13")
		return x
