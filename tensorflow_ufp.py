# coding: utf-8

# ### Code to accompany post [Universal Function Approximation using TensorFlow](http://deliprao.com/archives/100)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# A $40\times{40}$ network with two hidden layers gets to MSE $1.17$ after $2,300$ epochs, and never gets better.
# A network with one hidden layer of $40$ nodes gets to MSE $1.32$ after $10,000$ epochs, and is still improving.
# The $40\times{40}$ network achieves MSE $1.32$ somewhere between $1,000$ and $1,100$ epochs, when it is still
# improving rapidly.

# ### Find out about environments

# The following is not a reliable method, and doesn't port to Python 2.7, nor
# (possibly) in every sub-version of Python 3. It's really here just to tell us
# the name of the environment we're in, in the case we're missing some
# packages and need to troubleshoot the environment. See https://goo.gl/EXq7pE.

import sys
import os

if 'CONDA_DEFAULT_ENV' in os.environ:
    print(f"running in conda environment {os.environ['CONDA_DEFAULT_ENV']}")
else:
    print("not running in a conda environment")

if hasattr(sys, 'real_prefix'):
    print(f"sys has real_prefix {sys.real_prefix}")
else:
    print("sys has no real_prefix")

if hasattr(sys, 'base_prefix'):
    print(f"sys has base_prefix {sys.base_prefix}")
else:
    print("sys has no base_prefix")

if hasattr(sys, 'prefix'):
    print(f"sys has prefix {sys.prefix}")
else:
    print("sys has no prefix")


# ### Hyperparameters and constants

# The two-layer netword with 40 nodes in each layer achieves 1.41 MSE by epoch
# 1500, whereas, with 20 nodes in each layer, achieves only 2.03.

# In[4]:


np.random.seed(1000) # for repro

function_to_learn = lambda x: np.sin(x) + 0.1 * np.random.randn(*x.shape)

NUM_HIDDEN_NODES = 40

NUM_NODES_LAYER_1 = 40
NUM_NODES_LAYER_2 = 40

NUM_EXAMPLES = 1000
TRAIN_SPLIT = .8
MINI_BATCH_SIZE = 100
NUM_EPOCHS = 10000


def uniform_scrambled_col_vec(left, right, dim):
    return np.random.uniform(left, right, (1, dim)).T


all_x = uniform_scrambled_col_vec(-2 * np.pi, 2 * np.pi, NUM_EXAMPLES)
plt.plot(all_x.T[0]);


# Not sure why the original author shuffled the $x$ array. Looks like it was
# already shuffled.

# The original is now marked as a raw cell. Mark it 'y' for code if you want to
# execute it.
# np.random.shuffle(all_x)
# plt.plot(all_x.T[0])


def make_data_sets(xs, func, num_examples, train_split):
    train_size = int(num_examples * train_split)
    trainx = xs[:train_size]
    validx = xs[train_size:]
    return trainx, func(trainx), validx, func(validx)


trainx, trainy, validx, validy =     make_data_sets(
        all_x, 
        function_to_learn, 
        NUM_EXAMPLES, 
        TRAIN_SPLIT)


plt.figure(1)
plt.scatter(trainx, trainy, c='green', label='train')
plt.scatter(validx, validy, c='red', label='validation')
plt.legend()


X = tf.placeholder(tf.float32, [None, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")


def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    """Make a bunch of anonymous TF global variables that must later be
    initialized."""
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(
            tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else:  # xavier
        (fan_in, fan_out) = xavier_params
        blorp = 4 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.Variable(tf.random_uniform(
            shape, minval=-blorp, maxval=blorp, dtype=tf.float32))


the_model = None
the_model_2 = None

# test

def model_with_one_layer(X, num_hidden=10):
    global the_model
    w_h = init_weights([1, num_hidden], 'xavier', xavier_params=(1, num_hidden))
    b_h = init_weights([1, num_hidden], 'zeros')
    h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)
    
    w_o = init_weights([num_hidden, 1], 'xavier', xavier_params=(num_hidden, 1))
    b_o = init_weights([1, 1], 'zeros')
    the_model = tf.matmul(h, w_o) + b_o
    return the_model


def model_with_two_layers(X, num_h1=10, num_h2=10):
    global the_model_2
    w_h1 = init_weights([1, num_h1], 'xavier', xavier_params=(1, num_h1))
    b_h1 = init_weights([1, num_h1], 'zeros')
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1) + b_h1)
    
    w_h2 = init_weights([num_h1, num_h2], 'xavier', xavier_params=(num_h1, num_h2))
    b_h2 = init_weights([1, num_h2], 'zeros')
    h2 = tf.nn.sigmoid(tf.matmul(h1, w_h2) + b_h2)
    
    w_o = init_weights([num_h2, 1], 'xavier', xavier_params=(num_h2, 1))
    b_o = init_weights([1, 1], 'zeros')
    the_model_2 = tf.matmul(h2, w_o) + b_o
    return the_model_2


yhat = model_with_one_layer(X, NUM_HIDDEN_NODES)
yhat_2 = model_with_two_layers(X, NUM_NODES_LAYER_1, NUM_NODES_LAYER_2)


# In[15]:


train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(yhat - Y))
train_op_2 = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(yhat_2 - Y))


# In[16]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    errors = []
    for i in range(NUM_EPOCHS):
        for start, end in zip(range(0, len(trainx), MINI_BATCH_SIZE), 
                              range(MINI_BATCH_SIZE, len(trainx), MINI_BATCH_SIZE)):
            sess.run(train_op_2, feed_dict={X: trainx[start:end], Y: trainy[start:end]})
        mse = sess.run(tf.nn.l2_loss(yhat_2 - validy), feed_dict={X:validx})
        errors.append(mse)
        if i % 100 == 0: 
            result = sess.run(the_model_2, feed_dict={X:validx})
            plt.figure(1)
            plt.scatter(validx, validy, c='green', label='validation')
            plt.scatter(validx, result, c='red', label='eval(validx)')
            plt.legend()
            plt.show()
            print(f"epoch {i}, validation MSE {mse:6.2f}")
        
plt.xlabel('#epochs')
plt.ylabel('MSE')
plt.plot(errors);


# In[17]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    errors = []
    for i in range(NUM_EPOCHS):
        for start, end in zip(range(0, len(trainx), MINI_BATCH_SIZE), 
                              range(MINI_BATCH_SIZE, len(trainx), MINI_BATCH_SIZE)):
            sess.run(train_op, feed_dict={X: trainx[start:end], Y: trainy[start:end]})
        mse = sess.run(tf.nn.l2_loss(yhat - validy), feed_dict={X:validx})
        errors.append(mse)
        if i % 100 == 0: 
            result = sess.run(the_model, feed_dict={X:validx})
            plt.figure(1)
            plt.scatter(validx, validy, c='green', label='validation')
            plt.scatter(validx, result, c='red', label='eval(validx)')
            plt.legend()
            plt.show()
            print(f"epoch {i}, validation MSE {mse:6.2f}")
        
plt.xlabel('#epochs')
plt.ylabel('MSE')
plt.plot(errors);

