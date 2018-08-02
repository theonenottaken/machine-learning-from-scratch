import numpy as np
import math

def randomize_trainX_trainY(tX, tY):
	x_with_y = np.c_[tX, tY]
	np.random.shuffle(x_with_y)
	tX = x_with_y[:, :-1]
	tY = x_with_y[:, -1]
	return tX, tY

def sigmoid(x):
	if x < -500:
		return 0.0
	return (1.0 / (1.0 + np.exp(-1 * x)))

def softmax(vect):
	out_vect = [0] * len(vect)
	# get the sum of the exps of the whole array
	sum_exps = 0
	for x in vect:
		sum_exps = sum_exps + np.exp(x)
	for i in range(len(vect)):
		out_vect[i] = np.exp(vect[i]) / sum_exps
	return out_vect

def forward_propagation(params, inVect, truthVal, testing=True):

	w1 = params[0]
	b1 = params[1]
	w2 = params[2]
	b2 = params[3]

	z1 = np.dot(w1, inVect) + b1 	# vector of size H
	active_func = np.vectorize(sigmoid)
	h = active_func(z1)					# output of hidden layer, size H
	z2 = np.dot(w2, h) + b2 		# vector of size 10

	yhat = softmax(z2)				# normalizes z2 to a confidence level for each class

	if testing:
		vals = [z1, h, z2, yhat]

		length = len(yhat)
		truth_vect = [0] * length
		truth_vect[int(truthVal)] = 1
		sum_loss = 0
		for i in range(length):
			sum_loss = sum_loss + (truth_vect[i] * math.log(yhat[i]))
		return vals, sum_loss * (-1)
	else:
		return yhat

def back_propagation(params, x, y, vals):
	length = len(vals[3])
	truth_vect = [0] * length
	truth_vect[int(y)] = 1
	truth_vect = np.array(truth_vect)
	yhat = np.array(vals[3])
	h = np.array(vals[1])
	# compute dL/dW2
	diff = yhat - truth_vect
	diff = diff.reshape(len(diff), 1)
	dL_dw2 = np.dot(diff, np.transpose(h.reshape(len(h), 1)))			# (yhat - y) * h

	dL_db2 = diff									# (yhat - y) * 1

	w2 = params[2]
	dL_dh = np.dot(np.transpose(w2), diff)

	z1 = vals[0]
	sig_Z1 = np.vectorize(sigmoid)(vals[0])
	sig_Z1 = sig_Z1.reshape(len(sig_Z1), 1)
	ones_H = np.ones(len(z1)).reshape(len(z1), 1)
	one_minus_sig_Z1 = ones_H - sig_Z1
	one_minus_sig_Z1 = one_minus_sig_Z1.reshape(len(one_minus_sig_Z1), 1)
	sig_prod = sig_Z1 * one_minus_sig_Z1

	product = dL_dh * sig_prod
	dL_dw1 = np.dot(product, np.transpose(x.reshape(len(x), 1)))

	dL_db1 = product

	return [dL_dw1, dL_db1, dL_dw2, dL_db2]

def update_weights(params, grads, lr):
	length = 0
	for i in range(len(params)):
		vect = False
		if (len(np.shape(params[i])) < 2):
			vect = True
			length = len(params[i])
			params[i] = params[i].reshape(length, 1)
		new_weight = params[i] - (lr * grads[i])
		if vect:
			new_weight = new_weight.reshape(length)
		params[i] = new_weight
	return params

def validate(params, valid_x, valid_y):
	sum_of_losses = 0.0
	hits = 0
	for x,y in zip(valid_x, valid_y):
		vals, curr_loss = forward_propagation(params, x, y)
		sum_of_losses += curr_loss
		output = np.asarray(vals[3])
		if output.argmax() == y:
			hits += 1
	accuracy = float(hits) / float(np.shape(valid_x)[0])
	avg_loss = sum_of_losses / np.shape(valid_x)[0]
	return avg_loss, accuracy 

# text file contains 55,000 examples. For speed and ease, I only train on 20,000 of them.
train_x = np.loadtxt("train_x")
train_x = train_x * (1.0 / 255.0)
train_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")

SIZE = np.shape(train_x)[0]

# shuffle  the matrices to ensure good training data when we split 80:20 training and validation
train_x, train_y = randomize_trainX_trainY(train_x, train_y)

train_size = int(SIZE * 0.8)

valid_x = train_x[train_size + 1:, :]
valid_y = train_y[train_size + 1:]
train_x = train_x[:train_size + 1, :]
train_y = train_y[:train_size + 1]

#intitialize parameters to random values to start
H = 300
learning_rate = 0.01
num_epochs = 20

in_size = 784	# the size of a single training example
out_size = 10   	# number of classes

weight1 = np.random.rand(H, in_size)
weight1 = weight1 * .16 - 0.08
bias1 = np.random.rand(H)
bias1 = bias1 * 0.6 - 0.3
weight2 = np.random.rand(out_size, H)
weight2 = weight2 * .16 - 0.08
bias2 = np.random.rand(out_size)
bias2 = bias2 * 0.6 - 0.3
params = [weight1, bias1, weight2, bias2]

# train the neural network

for i in range(num_epochs):
	sum_of_losses = 0.0
	train_x, train_y = randomize_trainX_trainY(train_x, train_y)
	# train on the 20,000 examples
	for x,y in zip(train_x, train_y):
		vals, loss = forward_propagation(params, x, y)
		sum_of_losses = sum_of_losses + loss
		gradients = back_propagation(params, x, y, vals)
		params = update_weights(params, gradients, learning_rate)
	# validate and collect data on how well we did on the validation set
	validation_loss, accuracy = validate(params, valid_x, valid_y)
	avg_loss = sum_of_losses / np.shape(train_x)[0]
	print i, avg_loss, validation_loss, "{}%".format(accuracy * 100)
