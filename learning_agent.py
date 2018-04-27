import numpy as np
import tensorflow as tf
import time
from util import *
from gamelogic import *

class QLearningAgent(object):
    """An agent that learns by Reinforcement Learning"""
    def __init__conv(self):
        self.n_filters = 256
        self.n_filters2 = 512
        self.input = tf.placeholder(shape=[None,4,4,16], dtype=tf.float32)
        self.conv_a = tf.layers.conv2d(self.input, self.n_filters, (2,1), padding='valid', activation=tf.nn.relu)
        self.conv_b = tf.layers.conv2d(self.input, self.n_filters, (1,2), padding='valid', activation=tf.nn.relu)
        self.conv_aa = tf.layers.conv2d(self.conv_a, self.n_filters2, (2,1), padding='valid', activation=tf.nn.relu)
        self.conv_ab = tf.layers.conv2d(self.conv_a, self.n_filters2, (1,2), padding='valid', activation=tf.nn.relu)
        self.conv_ba = tf.layers.conv2d(self.conv_b, self.n_filters2, (2,1), padding='valid', activation=tf.nn.relu)
        self.conv_bb = tf.layers.conv2d(self.conv_b, self.n_filters2, (1,2), padding='valid', activation=tf.nn.relu)

        all_conv_layers = [self.conv_a, self.conv_b, self.conv_aa, self.conv_ab, self.conv_ba, self.conv_bb]

        self.conv_flat_concat = tf.concat([tf.layers.flatten(x) for x in all_conv_layers], 1)
        self.dense = tf.layers.dense(self.conv_flat_concat, 200, activation=tf.nn.relu)

        self.Qout = tf.layers.dense(self.dense, 4)
        self.get_action = tf.argmax(self.Qout,1)

    def __init__(self, model_dest=None):
        self.__init__conv()
        # this was 0.0001
        self.lr = 0.00005
        self.model_dest = model_dest
        # loss op
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        # optimizer
        # optimizer = tf.train.MomentumOptimizer(self.lr, .95)
        optimizer = tf.train.AdamOptimizer(self.lr)
        # training op
        self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

    def play(self):
        if not self.model_dest:
            sys.exit('Error: No path to model provided')

        sess = tf.Session()
        saver = tf.train.Saver()

        print('Restoring model from %s...' % self.model_dest)
        saver.restore(sess, self.model_dest)
        print('Done.')

        board = GameBoard()
        actions = range(4)
        commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}

        while not board.is_game_over():
            a,allQ = sess.run([self.get_action,self.Qout],feed_dict={self.input: convert_state(board.grid).reshape((1,4,4,16))})

            a = a[0]
            allQ = allQ[0]
            a = filter_legal_actions(a, allQ, board)
            
            commands[a]()
            board.add_tile()
            print(board.grid)

class SupervisedLearningAgent(object):
    """An agent that learns from an advanced heuristic-search AI"""
    def __init__conv(self):
        self.n_filters = 256
        self.n_filters2 = 512
        self.input = tf.placeholder(shape=[None,4,4,16], dtype=tf.float32)
        self.conv_a = tf.layers.conv2d(self.input, self.n_filters, (2,1), padding='valid', activation=tf.nn.relu)
        self.conv_b = tf.layers.conv2d(self.input, self.n_filters, (1,2), padding='valid', activation=tf.nn.relu)
        self.conv_aa = tf.layers.conv2d(self.conv_a, self.n_filters2, (2,1), padding='valid', activation=tf.nn.relu)
        self.conv_ab = tf.layers.conv2d(self.conv_a, self.n_filters2, (1,2), padding='valid', activation=tf.nn.relu)
        self.conv_ba = tf.layers.conv2d(self.conv_b, self.n_filters2, (2,1), padding='valid', activation=tf.nn.relu)
        self.conv_bb = tf.layers.conv2d(self.conv_b, self.n_filters2, (1,2), padding='valid', activation=tf.nn.relu)

        all_conv_layers = [self.conv_a, self.conv_b, self.conv_aa, self.conv_ab, self.conv_ba, self.conv_bb]
        # all_conv_layers = [self.conv_a, self.conv_b]

        self.conv_flat_concat = tf.concat([tf.layers.flatten(x) for x in all_conv_layers], 1)
        self.dense = tf.layers.dense(self.conv_flat_concat, 200, activation=tf.nn.relu)
        # self.dense = tf.layers.dense(self.conv_flat_concat, 300, activation=tf.nn.relu)

        self.logits = tf.layers.dense(self.dense, 4)
        self.get_action = tf.argmax(self.logits,1)

    def __init__deep(self):
        self.input = tf.placeholder(shape=[None,4,4,16], dtype=tf.float32)
        self.l1 = tf.layers.dense(tf.layers.flatten(self.input), 900, activation=tf.nn.relu)
        self.l2 = tf.layers.dense(self.l1, 300, activation=tf.nn.relu)
        self.l3 = tf.layers.dense(self.l2, 200, activation=tf.nn.relu)

        self.logits = tf.layers.dense(self.l3, 4)
        self.get_action = tf.argmax(self.logits,1)

    def __init__(self, model_dest=None):
        self.__init__conv()
        # self.__init__deep()
        self.lr = 0.0005
        self.model_dest = model_dest

        self.labels = tf.placeholder(shape=[None,], dtype=tf.int32)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
        # regularization
        # self.train_loss_op = tf.losses.mean_squared_error(labels=self.nextQ, predictions=self.Qout)
        l1_regularizer = tf.contrib.layers.l1_regularizer(
           scale=0.000000, scope=None
        )
        weights = tf.trainable_variables() # all vars of the graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        self.loss = self.loss + regularization_penalty
        # optimizer
        # optimizer = tf.train.MomentumOptimizer(self.lr, .95)
        optimizer = tf.train.AdamOptimizer(self.lr)
        # training op
        self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

    def play(self):
        if not self.model_dest:
            sys.exit('Error: No path to model provided')

        sess = tf.Session()
        saver = tf.train.Saver()

        print('Restoring model from %s...' % self.model_dest)
        saver.restore(sess, self.model_dest)
        print('Done.')

        board = GameBoard()
        actions = range(4)
        commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}

        while not board.is_game_over():
            a = sess.run([self.get_action],feed_dict={self.input: convert_state(board.grid).reshape((1,4,4,16))})

            a = a[0][0]
            allQ = np.random.random(4)
            allQ[a] = 2.0
            a = filter_legal_actions(a, allQ, board)
            
            commands[a]()
            board.add_tile()
            print(board.grid)

