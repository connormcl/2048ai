import numpy as np
from copy import deepcopy
from gamelogic import *

# n: integer
# returns: 16-bit binary number, each bit an entry in an numpy array
def binary_vector(n):
	n = int(n)
	if n == 0:
		return np.array(list("{0:016b}".format(1)), dtype=int)
	else:
		return np.array(list("{0:016b}".format(n)), dtype=int)

# # grid: raw GameBoard grid
# # returns: 1x256 binary encoding of grid
# def convert_state(grid):
# 	grid = deepcopy(grid)
# 	flat = grid.flatten()
# 	out = binary_vector(flat[0])
# 	for i in range(1,len(flat)):
# 		out = np.append(out,binary_vector(flat[i]))
# 	return out

# grid: raw GameBoard grid
# returns: 1x256 binary encoding of grid
def convert_state(grid):
	grid = deepcopy(grid)
	out = np.zeros((4,4,16))
	for i in range(4):
		for j in range(4):
			out[i,j] = binary_vector(grid[i,j])
	return out

def normalize_state(s):
	return s/16

def normalize_reward(r):
	return max(0,np.log2(r))/16

def update_target_graph(tfVars, tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars//2]):
		op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
	return op_holder

def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def empty_heuristic(board, next_board):
	return np.count_nonzero(board.grid) - np.count_nonzero(next_board.grid)

def corner_heuristic(board, next_board):
	if (next_board.largest_tile > board.largest_tile) and (next_board.largest_tile > 128):
		if (next_board.largest_tile == next_board.grid[0,0] or next_board.largest_tile == next_board.grid[0,3] or \
			next_board.largest_tile == next_board.grid[3,0] or next_board.largest_tile == next_board.grid[3,3]):
			# return 5 * (np.log2(next_board.largest_tile) - np.log2(128))
			return 8 * (np.log2(next_board.largest_tile) - np.log2(128))
			# return next_board.largest_tile
		# else:
		# 	return np.log2(next_board.largest_tile) - np.log2(128)
		# return 1.0
	return 0.0
	

def compute_reward(board, next_board, d):
	# return 0.1 * empty_heuristic(board, next_board) + corner_heuristic(board, next_board)
	ch = corner_heuristic(board, next_board)
	# eh = 0.1 * empty_heuristic(board, next_board)
	return ch

def filter_legal_actions(a, allQ, board):
    if board.action_is_valid(a):
        return a
    bestA = None
    bestQ = -np.inf
    for i in range(len(allQ)):
        if allQ[i] > bestQ and board.action_is_valid(i):
            bestA = i
            bestQ = allQ[i]
    return bestA

def process_board(board):
	return convert_state(board.grid).reshape((1,4,4,16))

def process_grid(grid):
	return convert_state(grid).reshape((1,4,4,16))

def process_grid_batch(batch):
	new_batch = np.zeros((batch.shape[0], 4, 4, 16))
	for i in range(batch.shape[0]):
		new_batch[i] = process_grid(batch[i])
	return new_batch

# takes an action and a board, returns next_board and reward
def execute_and_observe(a, board):
	# copy the current board
    next_board = deepcopy(board)
    commands = {0: next_board.up, 1: next_board.down, 2: next_board.left, 3: next_board.right}
    # execute the chosen move
    commands[a]()
    # randomly add a new tile to the board
    next_board.add_tile()
    # see if the game is over
    d = next_board.is_game_over()
    # get the reward from executing the chosen move
    r = compute_reward(board, next_board, d)
    return next_board, r, d


class Experience(object):
	"""simple wrapper object for an experience tuple"""
	def __init__(self, board_t, a_t, r_t, board_t_plus_one):
		self.board_t = board_t.reshape((1,4,4,16))
		self.a_t = a_t
		self.r_t = r_t
		self.board_t_plus_one = board_t_plus_one.reshape((1,4,4,16))

	def unpack(self):
		return self.board_t, self.a_t, self.r_t, self.board_t_plus_one

class ExperienceReplayBuffer(object):
	"""helper class for experience replay functionality"""
	def __init__(self, max_size=100):
		self.max_size = max_size
		self.size = 0
		self.idx = 0
		self.boards = np.zeros((max_size, 4, 4, 16))
		self.actions = np.zeros((max_size,))
		self.rewards = np.zeros((max_size,))
		self.next_boards = np.zeros((max_size, 4, 4, 16))
		self.done_mask = np.zeros((max_size,))

	def add(self, board, a, r, next_board, d):
		if not self.idx < self.max_size:
			self.idx = 0
		self.boards[self.idx] = convert_state(board.grid).reshape((1,4,4,16))
		self.actions[self.idx] = a
		self.rewards[self.idx] = r
		self.next_boards[self.idx] = convert_state(next_board.grid).reshape((1,4,4,16))
		self.done_mask[self.idx] = d
		self.idx += 1
		if self.size < self.max_size:
			self.size += 1

	def add_weighted(self, board, a, r, next_board, d):
		if r == 0:
			n = 1
		else:
			n = int(r/2.0)
		for i in range(n):
			if not self.idx < self.max_size:
				self.idx = 0
			self.boards[self.idx] = convert_state(board.grid).reshape((1,4,4,16))
			self.actions[self.idx] = a
			self.rewards[self.idx] = r
			self.next_boards[self.idx] = convert_state(next_board.grid).reshape((1,4,4,16))
			self.done_mask[self.idx] = d
			self.idx += 1
			if self.size < self.max_size:
				self.size += 1

	# return boards (batch_size, 4, 4, 16), qVals (batch_size, 4), rewards (batch_size, 1) next_boards (batch_size, 4, 4, 16)
	def get_batch(self, batch_size=5):
		# make sure we have enough entries for the batch
		if batch_size > self.size:
			batch_size = self.size
		# compute random indices
		random_indices = np.arange(self.size)
		np.random.shuffle(random_indices)
		random_indices = random_indices[0:batch_size]
		# use random indices to grab batch
		boards = self.boards[random_indices]
		actions = self.actions[random_indices]
		rewards = self.rewards[random_indices]
		next_boards = self.next_boards[random_indices]
		done_mask = self.done_mask[random_indices]

		return boards, actions, rewards, next_boards, done_mask

	# def debug(self):
	# 	print(self.experiences)
	# 	print(self.rewards)
	# 	print(self.idx)
	# 	print(self.probs())
	# 	print(sum(self.rewards))

	# def probs(self):
	# 	r = np.array(self.rewards)
	# 	return r / r.sum()
	# 	# totReward = sum(self.rewards)
	# 	# return [r / totReward for r in self.rewards]

	# def add(self, e, r):
	# 	r = 16 - r
	# 	if len(self.experiences) < self.max_size:
	# 		self.experiences.append(e)
	# 		self.rewards.append(r)
	# 	else:
	# 		if self.idx >= self.max_size:
	# 			self.idx = 0

	# 		self.experiences[self.idx] = e
	# 		self.rewards[self.idx] = r
	# 		self.idx += 1

			
	def uniform_sample(self, n=30):
		if n > len(self.experiences):
			n = len(self.experiences)
		return np.random.choice(self.experiences, size=n, replace=False)

	def weighted_sample(self, n=30):
		if n > len(self.experiences):
			n = len(self.experiences)
		# import pdb ; pdb.set_trace()
		return np.random.choice(self.experiences, p=list(self.probs()), size=n, replace=True)
