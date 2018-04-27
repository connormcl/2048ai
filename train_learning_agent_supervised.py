import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from learning_agent import SupervisedLearningAgent
from util import *
from gamelogic import *
import time

def parse_board(x):
	x = x.split('\n')[1:5]
	try:
		x[3] = x[3].split('\n')[0]
	except Exception as e:
		import pdb ; pdb.set_trace()
		raise e
	
	for i in range(4):
		x[i] = [int(y) for y in x[i].split()]
	return np.array(x)

def parse_move(x):
	return int(x.split('Take move: ')[1].split('\n')[0])

# main parse function
def get_game_data():
	boards = []
	moves = []
	num_training_games = 4027
	for i in range(num_training_games+1):
		# import pdb ; pdb.set_trace()
		game_log = 'game_logs/game' + str(i) + '.log'
		f = open(game_log, 'r')

		print('Parsing', game_log, '...')

		text = f.read()

		games = text.split('#')
		games = games[1:]

		for chunk in games:
			board = parse_board(chunk)
			move = parse_move(chunk)
			# parsed.append((board, move))
			boards.append(board)
			moves.append(move)
		print('Done.')

	return boards, moves

def train(X_train, y_train, X_test, y_test, num_epochs=500, batch_size=100, model_dest=None, restore_model=False):
	tf.reset_default_graph()
	net = SupervisedLearningAgent()

	sess = tf.Session()
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	sess.run(init)

	if model_dest and restore_model:
	    print('Restoring model from %s...' % model_dest)
	    saver.restore(sess, model_dest)
	    print('Done.')

	for epoch in range(num_epochs):
		t0 = time.time()
		# train batch
		random_indices = np.arange(X_train.shape[0])
		np.random.shuffle(random_indices)
		batch = (X_train[random_indices[0:batch_size]], y_train[random_indices[0:batch_size]])
		# test batch
		random_indices = np.arange(X_test.shape[0])
		np.random.shuffle(random_indices)
		test_batch = (X_test[random_indices[0:batch_size]], y_test[random_indices[0:batch_size]])

		# import pdb ; pdb.set_trace()
		
		# gradient update step
		# sess.run(net.train_op, feed_dict={net.input: batch[0].reshape((batch_size,4,4,16)), net.labels: batch[1]})
		sess.run(net.train_op, feed_dict={net.input: process_grid_batch(batch[0]), net.labels: batch[1]})

		# accuracies on current batch
		# train_accuracy = np.mean(batch[1] == sess.run(net.get_action, feed_dict={net.input: batch[0].reshape((1,4,4,16)), net.labels: batch[1]}))
		# test_accuracy = np.mean(test_batch[1] == sess.run(net.get_action, feed_dict={net.input: test_batch[0].reshape((1,4,4,16)), net.labels: test_batch[1]}))
		train_accuracy = np.mean(batch[1] == sess.run(net.get_action, feed_dict={net.input: process_grid_batch(batch[0]), net.labels: batch[1]}))
		test_accuracy = np.mean(test_batch[1] == sess.run(net.get_action, feed_dict={net.input: process_grid_batch(test_batch[0]), net.labels: test_batch[1]}))

		print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%% (%.2f seconds)"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy, time.time() - t0))
		if (epoch % 100 == 0) and model_dest:
		    print('Saving model to %s...' % model_dest)
		    saver.save(sess, model_dest)
		    print('Done.')




if __name__ == '__main__':
	X, y = get_game_data()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_test = np.array(X_test)
	y_test = np.array(y_test)
	train(X_train, y_train, X_test, y_test, num_epochs=10000, batch_size=1000, model_dest='/tmp/2048_supervised_model1.ckpt', restore_model=True)
	# train(X_train, y_train, X_test, y_test, num_epochs=10000, batch_size=500, model_dest='/tmp/2048_supervised_model2.ckpt', restore_model=True)
	# train(X_train, y_train, X_test, y_test, num_epochs=10000, batch_size=1000, model_dest='/tmp/2048_supervised_model3.ckpt', restore_model=False)
