from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np


class CifarData:
	def __init__(self, filenames):
		"""
		The initialization of the data class
		:param filenames:  The name of the file from which data needs to be imported
		"""
		all_data = []
		all_labels = []
		for filename in filenames:
			new_data, new_labels = self.load_data(filename)
			all_data.append(new_data)
			all_labels.append(new_labels)
		self._data = np.vstack(all_data)
		# Normalizing
		self._data = self._data / 255.
		self._labels = np.hstack(all_labels)
		self._num_data = self._data.shape[0]
		self._indicator = 0

	def load_data(self, filename):
		"""
		The function to load the CIFAR dataset
		:param filename: The name of the file from which data needs to be imported
		:return: Training dataset and testing dataset
		"""
		with open(filename, 'rb') as f:
			data = pickle.load(f, encoding = 'bytes')
			return data[b'data'], data[b'labels']

	def next_batch(self, batch_size):
		"""
		This function is used to return the next batch
		:param batch_size: The size of the batch we want
		:return: data for the current batch
		"""
		end_indictor = self._indicator + batch_size
		batch_data = self._data[self._indicator:end_indictor]
		batch_labels = self._labels[self._indicator:end_indictor]
		self._indicator = end_indictor
		return batch_data, batch_labels
