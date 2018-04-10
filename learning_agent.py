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
        self.lr = 0.0001
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
