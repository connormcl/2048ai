import tensorflow as tf
from learning_agent import SupervisedLearningAgent
from util import *
from gamelogic import *
from agents import RandomAgent, HeuristicAgent

agent = SupervisedLearningAgent(model_dest='/tmp/2048_supervised_model1.ckpt')

n_games = 100

sess = tf.Session()
saver = tf.train.Saver()

print('Restoring model from %s...' % agent.model_dest)
saver.restore(sess, agent.model_dest)
print('Done.')

tile_freqs1 = {2: 0, 4: 0, 8: 0, 16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}
tot_score1 = 0.0

for _ in range(n_games):
	board = GameBoard()
	actions = range(4)
	commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}

	while not board.is_game_over():
	    a = sess.run([agent.get_action],feed_dict={agent.input: convert_state(board.grid).reshape((1,4,4,16))})

	    a = a[0][0]
	    allQ = np.random.random(4)
	    allQ[a] = 2.0
	    a = filter_legal_actions(a, allQ, board)
	    
	    commands[a]()
	    board.add_tile()
	    # print(board.grid)

	tile_freqs1[int(board.largest_tile)] += 1
	tot_score1 += board.score
	print(int(board.largest_tile))

print('SupervisedLearningAgent')
print(tile_freqs1)
print('Average score:', str(tot_score1/n_games))

# heuristic agent
agent = HeuristicAgent()

tile_freqs2 = {2: 0, 4: 0, 8: 0, 16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}
tot_score2 = 0.0

for _ in range(n_games):
	board = GameBoard()
	actions = range(4)
	commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}

	while not board.is_game_over():
		a = agent.next_action(board.grid)
		commands[a]()
		board.add_tile()

	tile_freqs2[int(board.largest_tile)] += 1
	tot_score2 += board.score
	print(int(board.largest_tile))

print('HeuristicAgent')
print(tile_freqs2)
print('Average score:', str(tot_score2/n_games))

# random agent
agent = RandomAgent()

tile_freqs3 = {2: 0, 4: 0, 8: 0, 16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}
tot_score3 = 0.0

for _ in range(n_games):
	board = GameBoard()
	actions = range(4)
	commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}

	while not board.is_game_over():
		a = agent.next_action(board.grid)
		commands[a]()
		board.add_tile()

	tile_freqs3[int(board.largest_tile)] += 1
	tot_score3 += board.score
	print(int(board.largest_tile))

print('RandomAgent')
print(tile_freqs3)
print('Average score:', str(tot_score3/n_games))
print('SupervisedLearningAgent')
print(tile_freqs1)
print('Average score:', str(tot_score1/n_games))
print('HeuristicAgent')
print(tile_freqs2)
print('Average score:', str(tot_score2/n_games))


