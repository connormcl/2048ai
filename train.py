import numpy as np
import tensorflow as tf
from gamelogic import *
from copy import deepcopy
from util import *

def supervised(buffers):
	tf.reset_default_graph()

	# feed-forward part of network to choose actions
	inputs1 = tf.placeholder(shape=[1,256],dtype=tf.float32)
	# W = tf.Variable(tf.random_uniform([16,4],0,0.01))
	W1 = tf.Variable(tf.random_uniform([256,100],0,0.01))
	W2 = tf.Variable(tf.random_uniform([100,200],0,0.01))
	W3 = tf.Variable(tf.random_uniform([200,4],0,0.01))
	# h1 = tf.matmul(inputs1,W1) # 1x100
	# h2 = tf.matmul(h1,W2) # 1x200
	h1 = tf.nn.relu(tf.matmul(inputs1,W1)) # 1x100
	h2 = tf.nn.relu(tf.matmul(h1,W2)) # 1x200
	Qout = tf.matmul(h2,W3) # 1x4
	predict = tf.argmax(Qout,1)

	# loss by taking the sum of squares difference between target and prediction Q values
	nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	# optimizer = tf.train.AdamOptimizer(self.lr)
	# for a in range(4):
		

	# scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	gvs = optimizer.compute_gradients(loss) #, var_list=scope_vars)
	updateModel = optimizer.apply_gradients(gvs)
	# self.grad_norm = tf.global_norm(grads)

	# updateModel = optimizer.minimize(loss)

	init = tf.initialize_all_variables()

	y = .99
	epsilon = 0.1
	num_episodes = 100

	# convert this to full-fledged supervised learning (classification prob where 
	# true y val is the action HeuristicAgent chooses)

	with tf.Session() as sess:
		sess.run(init)

		for replay_buffer in buffers:
			for epoch in range(1000):
				batch = replay_buffer.uniform_sample()

				for e in batch:
			# for e in replay_buffer.experiences:
					board_t, a_t, r_t, board_t_plus_one = e.unpack()

					_,allQ = sess.run([predict,Qout],feed_dict={inputs1: convert_state(board_t.grid).reshape((1,256))})

					Q1 = sess.run(Qout,feed_dict={inputs1: convert_state(board_t_plus_one.grid).reshape((1,256))})

					maxQ1 = np.max(Q1)
					targetQ = allQ
					targetQ[0,a_t] = r_t + y*maxQ1

					_ = sess.run([updateModel],feed_dict={inputs1: convert_state(board_t.grid).reshape((1,256)),nextQ:targetQ})
				print('.',end='',flush=True)

		for i in range(num_episodes):
			print("---------------------------------------------------------------"*100)

			# import pdb ; pdb.set_trace()

			board = GameBoard()
			actions = range(4)
			commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}

			print(board.grid)
			while not board.is_game_over():
				a,allQ = sess.run([predict,Qout],feed_dict={inputs1: convert_state(board.grid).reshape((1,256))})
				a = a[0]
				allQ2 = deepcopy(allQ[0])
				while not board.action_is_valid(a):
					allQ2[a] = -1
					a = np.argmax(allQ2)

				if np.random.rand(1) < epsilon:
					a = np.random.choice(actions)
					while not board.action_is_valid(a):
						a = np.random.choice(actions)

				commands[a]() # execute action on board
				board.add_tile()
				print(board.grid)

def train():
	tf.reset_default_graph()

	# feed-forward part of network to choose actions
	inputs1 = tf.placeholder(shape=[1,256],dtype=tf.float32)
	# W = tf.Variable(tf.random_uniform([16,4],0,0.01))
	W1 = tf.Variable(tf.random_uniform([256,100],0,0.01))
	W2 = tf.Variable(tf.random_uniform([100,200],0,0.01))
	W3 = tf.Variable(tf.random_uniform([200,4],0,0.01))
	# h1 = tf.matmul(inputs1,W1) # 1x100
	# h2 = tf.matmul(h1,W2) # 1x200
	h1 = tf.nn.relu(tf.matmul(inputs1,W1)) # 1x100
	h2 = tf.nn.relu(tf.matmul(h1,W2)) # 1x200
	Qout = tf.matmul(h2,W3) # 1x4
	predict = tf.argmax(Qout,1)

	# loss by taking the sum of squares difference between target and prediction Q values
	nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	# optimizer = tf.train.AdamOptimizer(self.lr)
	# for a in range(4):
		

	# scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	# import pdb ; pdb.set_trace()
	gvs = optimizer.compute_gradients(loss) #, var_list=scope_vars)
	# if self.config.grad_clip:
	#     new_gvs = list()
	#     for grad, var in gvs:
	#         if grad is not None:
	#             new_gvs.append((tf.clip_by_norm(grad, self.config.clip_val), var))
	#         else:
	#             new_gvs.append((grad, var))
	#     gvs = new_gvs
	#     # gvs = [(tf.clip_by_norm(grad, self.config.clip_val), var) for grad, var in gvs]
	# grads = [grad for grad, _ in gvs]
	# final part of .minimize(loss)
	updateModel = optimizer.apply_gradients(gvs)
	# self.grad_norm = tf.global_norm(grads)

	# updateModel = optimizer.minimize(loss)

	init = tf.initialize_all_variables()

	buffers = []
	from agents import HeuristicAgent
	agent = HeuristicAgent()
	# get experiences from HeuristicAgent
	num_games = 1
	for i in range(num_games):
		board = GameBoard()
		commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}
		largest_tile = 0
		replay_buffer = ExperienceReplayBuffer()
		while not board.is_game_over():
			s_t = deepcopy(board)
			a = agent.next_action(board.grid)
            #print('action:', a, ' |  score:', self.board.score)
			commands[a]()
			board.add_tile()
			replay_buffer.add(Experience(s_t, a, board.score, deepcopy(board)))
			if board.largest_tile > largest_tile:
				largest_tile = board.largest_tile
				print('Got', largest_tile, 'tile')
			# print(board.grid)
		buffers.append(replay_buffer)
		print('(%s/%s) games complete' % (i+1,num_games))

	supervised(buffers)

	print("---------------------------------------------------------------"*1000)

	y = .99
	epsilon = 0.1
	num_episodes = 100

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
				# import pdb ; pdb.set_trace()

				a,allQ = sess.run([predict,Qout],feed_dict={inputs1: convert_state(board.grid).reshape((1,256))})
				a = a[0]
				allQ2 = deepcopy(allQ[0])
				while not board.action_is_valid(a):
					allQ2[a] = -1
					a = np.argmax(allQ2)
				# a = int(a[0])
				# allQ = allQ[0]
				# while not board.action_is_valid(a):
				# 	allQ[a] = -1
				# 	a = np.argmax(allQ)

				if np.random.rand(1) < epsilon:
					a = np.random.choice(actions)
					while not board.action_is_valid(a):
						a = np.random.choice(actions)

				last_board = deepcopy(board)

				# score0 = board.score # score before action
				# prev_largest_tile = board.largest_tile
				commands[a]() # execute action on board
				# if board.largest_tile > prev_largest_tile and board.largest_tile > 500:
				# 	r = 1
				# else:
				# 	r = 0
				r = board.score
				# r = board.score - score0 # reward (increase in score)
				board.add_tile()
				print(board.grid)

				# store experience
				e = Experience(last_board, a, r, deepcopy(board))
				replay_buffer.add(e)
				# get batch of previous experiences
				batch = replay_buffer.uniform_sample()

				for e in batch:
					board_t, a_t, r_t, board_t_plus_one = e.unpack()

					Q1 = sess.run(Qout,feed_dict={inputs1: convert_state(board_t_plus_one.grid).reshape((1,256))})

					maxQ1 = np.max(Q1)
					targetQ = allQ
					targetQ[0,a_t] = r_t + y*maxQ1

					_ = sess.run([updateModel],feed_dict={inputs1: convert_state(board_t.grid).reshape((1,256)),nextQ:targetQ})

if __name__ == '__main__':
	train()

	# print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"