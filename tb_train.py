#!/usr/bin/python

import numpy as np
import tensorflow as tf
import readTBdata

import datetime
import copy

#data settings
posdata_filepath = "./data/pos.csv"
negdata_filepath = "./data/neg.csv"
irrdata_filepath = "./data/irr.csv"

smalltrain_mix_num = 15

#prepare data
starttime = datetime.datetime.now()

posdata = np.genfromtxt(posdata_filepath, delimiter=',')
perm = np.arange(len(posdata))
np.random.shuffle(perm)
posdata = posdata[perm]
	
negdata = np.genfromtxt(negdata_filepath, delimiter=',')
perm = np.arange(len(negdata))
np.random.shuffle(perm)
negdata = negdata[perm]
	
irrdata = np.genfromtxt(irrdata_filepath, delimiter=',')
perm = np.arange(len(irrdata))
np.random.shuffle(perm)
irrdata = irrdata[perm]
	
postrain = posdata[:len(posdata)/2]
postest = posdata[len(posdata)/2:]
poslabel_train = np.zeros((len(postrain),3))
poslabel_test = np.zeros((len(postest),3))
poslabel_train[:,0:1] = 1.0
poslabel_test[:,0:1] = 1.0

negtrain = negdata[:len(negdata)/2]
negtest = negdata[len(negdata)/2:]
neglabel_train = np.zeros((len(negtrain),3))
neglabel_test = np.zeros((len(negtest),3))
neglabel_train[:,2:3] = 1.0
neglabel_test[:,2:3] = 1.0

irrtrain = irrdata[:len(irrdata)/2]
irrtest = irrdata[len(irrdata)/2:]
irrlabel_train = np.zeros((len(irrtrain),3))
irrlabel_test = np.zeros((len(irrtest),3))
irrlabel_train[:,1:2] = 1.0
irrlabel_test[:,1:2] = 1.0

traindata = np.concatenate((postrain ,negtrain))
testdata = np.concatenate((postest ,negtest))
trainlabel = np.concatenate((poslabel_train, neglabel_train))
testlabel = np.concatenate((poslabel_test, neglabel_test))

smalltrain_data = postrain[:smalltrain_mix_num]
smalltrain_label = poslabel_train[:smalltrain_mix_num]
smalltrain_data = np.concatenate((smalltrain_data, irrtrain))
smalltrain_label = np.concatenate((smalltrain_label, irrlabel_train))

smalltest_data = postest[:smalltrain_mix_num]
smalltest_label = poslabel_test[:smalltrain_mix_num]
smalltest_data = np.concatenate((smalltest_data, irrtest))
smalltest_label = np.concatenate((smalltest_label, irrlabel_test))

# Parameters
learning_rate = 0.0004
training_epochs = 20 #45
batch_size = 10
display_step = 1
dropout_keep_prob = tf.placeholder("float")

# Network Parameters
n_hidden_1 = 512  # 1st layer number of features
n_hidden_2 = 512  # 2nd layer number of features
n_hidden_3 = 512 # 3rd layer
n_input = 4096	# data input size
n_classes = len(trainlabel[0])	# total classes: 3
n_sample_number = len(traindata)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# minibatch settings
epoch_completed = 0
rest_num_examples = n_sample_number
train_img_copy = -1
train_label_copy = -1
train_img_copy_rest = -1
train_label_copy_rest = -1
train_img_copy_new = -1
train_label_copy_new = -1

epoch_index = 0

train_img_copy = copy.copy(traindata)
train_label_copy = copy.copy(trainlabel)

def next_batch(batch_size):

	global epoch_index
	global epoch_completed
	global train_img_copy
	global train_label_copy
	global train_img_copy_rest
	global train_label_copy_rest
	global train_img_copy_new
	global train_label_copy_new

	start = epoch_index

	if epoch_completed == 0:
		perm0 = np.arange(n_sample_number)
		np.random.shuffle(perm0)
		train_img_copy = train_img_copy[perm0]
		train_label_copy = train_label_copy[perm0]
   #

	if start + batch_size > n_sample_number:
		epoch_completed += 1
		rest_num_examples = n_sample_number - start
		train_img_copy_rest = train_img_copy[start:n_sample_number]
		train_label_copy_rest = train_label_copy[start:n_sample_number]

		perm = np.arange(n_sample_number)
		np.random.shuffle(perm)
		train_label_copy = train_label_copy[perm]
		train_img_copy = train_img_copy[perm]
		start = 0
		epoch_index = batch_size - rest_num_examples
		end = epoch_index
		train_img_copy_new = train_img_copy[start:end]
		train_label_copy_new = train_label_copy[start:end]
		return np.concatenate((train_img_copy_rest, train_img_copy_new), axis=0), np.concatenate((train_label_copy_rest, train_label_copy_new), axis=0)
	else:
		epoch_index += batch_size
		end = epoch_index
		return train_img_copy[start:end], train_label_copy[start:end]
#

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

# Store layers weight & bias
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
# Construct model
pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Construct model (undersampled)
pred2 = multilayer_perceptron(x, weights2, biases2, dropout_keep_prob)
# Define loss and optimizer
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred2, labels=y))
optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost2)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	#"""
	# Training cycle - model 1
	for epoch in range(training_epochs):

		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#print "Accuracy:", accuracy.eval({x: data_train, y: label_train})
		#print "Accuracy:", accuracy.eval({x: data_test, y: label_test})

		avg_cost = 0.
		total_batch = int(n_sample_number / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = next_batch(batch_size)
			# Run optimization op (backprop) and cost op (to get loss value)
			feeds = {x: batch_x, y: batch_y, dropout_keep_prob: 0.6}
			sess.run(optimizer, feed_dict=feeds)
			feeds = {x: batch_x, y: batch_y, dropout_keep_prob: 1.0}
			c = sess.run(cost, feed_dict=feeds)
			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:

			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			# Calculate accuracy
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			test_accuracy = accuracy.eval({x: testdata, y: testlabel, dropout_keep_prob:1.0})
			train_accuracy = accuracy.eval({x: traindata, y: trainlabel, dropout_keep_prob:1.0})

			print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost)), " Train acc: ", train_accuracy, " Test acc: ", test_accuracy

	print("Optimization Finished!")
	
	endtime = datetime.datetime.now()
	time_passed = (endtime - starttime).seconds
	print "Time passed: ", time_passed
	
	# Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy (sensitivity):", accuracy.eval({x: testdata[:235], y: testlabel[:235], dropout_keep_prob:1.0}))
	print("Accuracy (specificity):", accuracy.eval({x: testdata[235:], y: testlabel[235:], dropout_keep_prob:1.0}))

	print ""
	print "======"
	print ""
	#"""
	#Model 2 var setting

	train_img_copy = copy.copy(smalltrain_data)
	train_label_copy = copy.copy(smalltrain_label)
	traindata = smalltrain_data 
	trainlabel = smalltrain_label
	testdata = smalltest_data
	testlabel = smalltest_label
	n_sample_number = len(traindata)

	# Training cycle - model 2 
	for epoch in range(training_epochs):

		correct_prediction = tf.equal(tf.argmax(pred2, 1), tf.argmax(y, 1))
		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#print "Accuracy:", accuracy.eval({x: data_train, y: label_train})
		#print "Accuracy:", accuracy.eval({x: data_test, y: label_test})

		avg_cost = 0.
		total_batch = int(n_sample_number / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = next_batch(batch_size)
			# Run optimization op (backprop) and cost op (to get loss value)
			feeds = {x: batch_x, y: batch_y, dropout_keep_prob: 0.6}
			sess.run(optimizer2, feed_dict=feeds)
			feeds = {x: batch_x, y: batch_y, dropout_keep_prob: 1.0}
			c = sess.run(cost2, feed_dict=feeds)
			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:

			correct_prediction = tf.equal(tf.argmax(pred2, 1), tf.argmax(y, 1))
			# Calculate accuracy
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			test_accuracy = accuracy.eval({x: testdata, y: testlabel, dropout_keep_prob:1.0})
			train_accuracy = accuracy.eval({x: traindata, y: trainlabel, dropout_keep_prob:1.0})

			print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost)), " Train acc: ", train_accuracy, " Test acc: ", test_accuracy

	print("Optimization Finished!")

	endtime = datetime.datetime.now()
	time_passed = (endtime - starttime).seconds
	print "Time passed: ", time_passed

	# Test model
	correct_prediction = tf.equal(tf.argmax(pred2, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy (sensitivity):", accuracy.eval({x: testdata[:15], y: testlabel[:15], dropout_keep_prob:1.0}))
	print("Accuracy (specificity):", accuracy.eval({x: tballdata.postest, y: tballdata.poslabel_test, dropout_keep_prob:1.0}))

	pred2 = tf.argmax(pred2,1)
	#print pred2.eval(feed_dict = {x: tballdata.irrtrain, y: tballdata.irrlabel_train, dropout_keep_prob:1.0})
	#print pred2.eval(feed_dict = {x: tballdata.irrtest, y: tballdata.irrlabel_test, dropout_keep_prob:1.0})
	#print np.sum(pred2.eval(feed_dict = {x: tballdata.postest, y: tballdata.poslabel_test, dropout_keep_prob:1.0}))

	saver.save(sess, "./model/twoNettrain_ckpt")	
	#"""
