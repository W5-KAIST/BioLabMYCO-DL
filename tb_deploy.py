#!/usr/bin/python

import numpy as np
import tensorflow as tf

#classification data
alldata = np.genfromtxt("./data/alldata.csv", delimiter=',')
IRR_FLAG = True

# Network Parameters
n_hidden_1 = 512  # 1st layer number of features
n_hidden_2 = 512  # 2nd layer number of features
n_hidden_3 = 512 # 3rd layer
n_input = 4096	# data input size
n_classes = 3	# total classes: 3
#n_sample_number = len(traindata)
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="bh1"),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name="bh2"),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]),name="bh3"),
	'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]),name="bho")
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1]),name="bb1"),
	'b2': tf.Variable(tf.random_normal([n_hidden_2]),name="bb2"),
	'b3': tf.Variable(tf.random_normal([n_hidden_3]),name="bb3"),
	'out': tf.Variable(tf.random_normal([n_classes]),name="bbo")
}

weights2 = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="sh1"),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name="sh2"),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]),name="sh3"),
	'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]),name="so")
}

biases2 = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1]),name="sb1"),
	'b2': tf.Variable(tf.random_normal([n_hidden_2]),name="sb2"),
	'b3': tf.Variable(tf.random_normal([n_hidden_3]),name="sb3"),
	'out': tf.Variable(tf.random_normal([n_classes]),name="sbo")
}

def multilayer_perceptron(x, weights, biases, keep_prob):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	# layer 3
	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	layer_3 = tf.nn.relu(layer_3)
	layer_3 = tf.nn.dropout(layer_3, keep_prob)
	# Output layer with linear activation
	out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
	return out_layer
#
dropout_keep_prob = tf.placeholder("float")
pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)
pred2 = multilayer_perceptron(x, weights2, biases2, dropout_keep_prob)

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, "./model/twoNettrain_ckpt")

	# Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Classify
	pred = tf.argmax(pred, 1)
	pred2 = tf.argmax(pred2,1)
	for index, i in enumerate(alldata):
		i = np.array([i])
		inf_1 = pred.eval(feed_dict = {x:i, dropout_keep_prob:1.0})[0]
		inf_2 = pred2.eval(feed_dict = {x:i, dropout_keep_prob:1.0})[0]

		if not IRR_FLAG:
			if inf_1 == 2:
				print (index, " Negative")
			if inf_1 == 0:
				print (index, " Positive")
	
		else:

			if inf_1 == 2:
				print (index, " Negative")
			if inf_1 == 0:
				if inf_2 == 0:
					print (index, " Positive")
				else:
					print (index, " Irregular")
