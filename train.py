import numpy as np
import tensorflow as tf
from gamelogic import *
from copy import deepcopy

def binary_vector(n):
	n = int(n)
	if n == 0:
		return np.array(list("{0:016b}".format(1)), dtype=int)
	else:
		return np.array(list("{0:016b}".format(n)), dtype=int)

def convert_state(grid):
	grid = deepcopy(grid)
	flat = grid.flatten()
	out = binary_vector(flat[0])
	for i in range(1,len(flat)):
		out = np.append(out,binary_vector(flat[i]))
	return out

def train():
	tf.reset_default_graph()

	# feed-forward part of network to choose actions
	inputs1 = tf.placeholder(shape=[1,256],dtype=tf.float32)
	# W = tf.Variable(tf.random_uniform([16,4],0,0.01))
	W1 = tf.Variable(tf.random_uniform([256,100],0,0.01))
	W2 = tf.Variable(tf.random_uniform([100,200],0,0.01))
	W3 = tf.Variable(tf.random_uniform([200,4],0,0.01))
	h1 = tf.matmul(inputs1,W1) # 1x100
	h2 = tf.matmul(h1,W2) # 1x200
	Qout = tf.matmul(h2,W3) # 1x4
	predict = tf.argmax(Qout,1)

	# loss by taking the sum of squares difference between target and prediction Q values
	nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	updateModel = trainer.minimize(loss)

	init = tf.initialize_all_variables()

	y = .99
	e = 0.1
	num_episodes = 2000
	actions = ['up','down','left','right']

	jList = []
	rList = []
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_episodes):
			rAll = 0
			# print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
			print("---------------------------------------------------------------")
			print("---------------------------------------------------------------")
			print("---------------------------------------------------------------")
			print("---------------------------------------------------------------")

			# import pdb ; pdb.set_trace()

			board = GameBoard()
			actions = range(4)
			commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}

			print(board.grid)
			while not board.is_game_over():
				# a = agent.next_action(board.grid)
				a,allQ = sess.run([predict,Qout],feed_dict={inputs1: convert_state(board.grid).reshape((1,256))})
				if np.random.rand(1) < e:
					a[0] = np.random.choice(actions)

				last_board_grid = deepcopy(board.grid)

				# score0 = board.score # score before action
				# prev_largest_tile = board.largest_tile
				commands[a[0]]() # execute action on board
				# if board.largest_tile > prev_largest_tile and board.largest_tile > 500:
				# 	r = 1
				# else:
				# 	r = 0
				r = board.score
				# r = board.score - score0 # reward (increase in score)
				board.add_tile()
				print(board.grid)

				Q1 = sess.run(Qout,feed_dict={inputs1: convert_state(board.grid).reshape((1,256))})

				maxQ1 = np.max(Q1)
				targetQ = allQ
				targetQ[0,a[0]] = r + y*maxQ1

				_ = sess.run([updateModel],feed_dict={inputs1: convert_state(last_board_grid).reshape((1,256)),nextQ:targetQ})
				rAll += r
			
			rList.append(rAll)

if __name__ == '__main__':
	train()

	# print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"