import numpy as np
import tensorflow as tf
from learning_agent import QLearningAgent
from util import *
from gamelogic import *
import time

def get_epsilon(startE, endE, pretrain_steps, annealing_steps, total_steps):
	if total_steps < pretrain_steps:
		return startE
	if total_steps > (pretrain_steps + annealing_steps):
		return endE
	step_drop = (startE - endE) / annealing_steps
	return startE - ((total_steps - pretrain_steps) * step_drop)

def train(num_episodes=500, batch_size=10, model_dest=None, restore_model=False):
	tf.reset_default_graph()
	mainQN = QLearningAgent()
	targetQN = QLearningAgent()
	
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	trainables = tf.trainable_variables()

	tau = 0.001
	targetOps = update_target_graph(trainables, tau)

	replay_buffer = ExperienceReplayBuffer(100)

	sess = tf.Session()
	sess.run(init)

	if model_dest and restore_model:
	    print('Restoring model from %s...' % model_dest)
	    saver.restore(sess, model_dest)
	    print('Done.')

	y = .99 # discount factor on the target Q-values
	epsilon = 0.1 # prob of random action
	update_freq = 4 # how often to perform a training step
	pretrain_steps = 10000 # how many random steps to take before training begins
	annealing_steps = 10000 # how many steps of random actions before training begins
	total_steps = 0
	# total_steps = pretrain_steps + 1000
	# startE = 1.0
	startE = 0.1
	endE = 0.1
	all_losses = []

	for i in range(num_episodes):
		t0 = time.time()
		board = GameBoard()
		actions = range(4)
		largest_tile = 2
		total_reward = 0
		losses = []

		pick_move_time = 0.0
		add_experience_time = 0.0
		grad_descent_time = 0.0

		while not board.is_game_over():
			# import pdb ; pdb.set_trace()
			t1 = time.time()
			# epsilon-greedy policy: pick a random action with probability epsilon
			if np.random.rand(1) < get_epsilon(startE, endE, pretrain_steps, annealing_steps, total_steps):
				a = np.random.choice(actions)
				# while not board.action_is_valid(a):
					# a = np.random.choice(actions)
			else:
				# given current board state, get agent's next action and Q-values of all actions
				a,allQ = sess.run([mainQN.get_action,mainQN.Qout],feed_dict={mainQN.input: process_board(board)})
				# unpack action and Q-values
				a = a[0]
				allQ = allQ[0]
				# choose the best valid action
				a = filter_legal_actions(a, allQ, board)

			# run action on the board, observe the next board state and reward
			next_board, r, done = execute_and_observe(a, board)
			total_steps += 1
			total_reward += r

			pick_move_time += time.time() - t1
			t2 = time.time()

			# add this experience to the replay buffer
			# e = Experience(last_board, allQ, a, r, deepcopy(board))
			# if (r > 0) or (total_steps < pretrain_steps):
			replay_buffer.add(board, a, r, next_board, done)
			# replay_buffer.add_weighted(board, a, r, next_board, done)
			board = deepcopy(next_board)

			add_experience_time += time.time() - t2
			t3 = time.time()

			if (total_steps % update_freq == 0) and total_steps > pretrain_steps:
				# import pdb ; pdb.set_trace()
	            # sample experiences from replay buffer
	            # batch = replay_buffer.uniform_sample(n=batch_size)
				boards, actions, rewards, next_boards, done_mask = replay_buffer.get_batch(batch_size)

	            # get actions from main Q network
				Q1 = sess.run(mainQN.get_action,feed_dict={mainQN.input: next_boards})

	            # get Q-values from target Q network
				Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.input: next_boards})

				# 0 for game over, 1 for game active
				end_multiplier = -(done_mask - 1)
	            # instead of max of posible Q-vals, grab Q-vals from targetQN
				doubleQ = Q2[range(batch_size),Q1]
				targetQ = rewards + (y * doubleQ * end_multiplier)

	            # gradient descent step on mainQN
				_, loss = sess.run([mainQN.train_op, mainQN.loss], feed_dict={mainQN.input: boards, mainQN.targetQ: targetQ, mainQN.actions: actions})
				losses.append(loss)
	            # update targetQN towards mainQN (at rate specified by tau)
				update_target(targetOps, sess)
			grad_descent_time += time.time() - t3
		avg_loss = 0.0
		if losses:
			avg_loss = sum(losses)/len(losses)
			all_losses.append(avg_loss)
		# print('(%s/%s) games complete (%.2f seconds; pick move: %.2f, add experience: %.2f, grad descent: %.2f) | Total reward = %.2f | Avg loss = %.2f'
		# 		% (i+1,num_episodes, time.time() - t0, pick_move_time, add_experience_time, grad_descent_time, total_reward, avg_loss))
		print('(%s/%s) games complete (%.2f sec; grad desc: %.2f) | epsilon = %.4f | tot rew = %.2f | best tile = %d | avg loss = %.2f'
				% (i+1,num_episodes, time.time() - t0, grad_descent_time, get_epsilon(startE, endE, pretrain_steps, annealing_steps, total_steps), total_reward, board.largest_tile, avg_loss))

		if (i % 100 == 0) and model_dest:
		    print('Saving model to %s...' % model_dest)
		    saver.save(sess, model_dest)
		    print('Done.')

	print('Total steps: %s' % total_steps)

	# import matplotlib
	# matplotlib.use('Agg')
	# import matplotlib.pyplot as plt
	# fig = plt.figure()
	# plt.plot(all_losses)
	# fig.savefig('loss_plot.png')

	# import matplotlib.pyplot as plt
	# plt.plot(all_losses)
	# plt.show()

	sess.close()


if __name__ == '__main__':
	train(num_episodes=50000, batch_size=32, model_dest='/tmp/2048_model1.ckpt', restore_model=True)
