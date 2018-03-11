from gamelogic import *
import numpy as np

class RandomAgent(object):
    """An agent that selects its moves randomly"""
    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']

    def next_action(self, state):
        return np.random.choice(self.actions)

class HeuristicAgent(object):
    """An agent that searches for the best move according to its heuristic functions"""
    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']
        self.max_depth = 4
        # self.discount = 0.7
        self.discount = 0.7

    def corner_heuristic(self, state, last_state):
        highest_tile = np.max(state)
        corners = [state[0,0], state[0,3], state[3,0], state[3,3]]
        last_corners = [last_state[0,0], last_state[0,3], last_state[3,0], last_state[3,3]]
        if highest_tile in corners:
            for i in range(len(corners)):
                if highest_tile == corners[i]:
                    if highest_tile == last_corners[i]:
                        return highest_tile
                    else:
                        return highest_tile * 0.5
        return 0

    def highest_merged_heuristic(self, state):
        flat = state.flatten()
        flat.sort()
        if len(flat) > 2 and flat[0] == flat[1]:
            return -flat[0] / 2.0
        else:
            return 0

    # def corner_switch_heuristic(self, state, last_state):
    #     if np.max(state) == np.max(last_state):
    #         highest_tile = np.max(state)
    #         score = 0
    #         corners = [(0,0), (0,3), (3,0), (3,3)]
    #         corner_vals = [state[0,0], state[0,3], state[3,0], state[3,3]]
    #         if highest_tile in corner_vals:
    #             for i in range(len(corner_vals)):
    #                 if flat[0] == corner_vals[i]:
    #                     corner = corners[i]



    def ith_from_corner(self, state, corner, i, ith):
        if corner == (0,0):
            return (state[i,0] == ith) or (state[0,i] == ith)
        elif corner == (0,3):
            return (state[0,3-i] == ith) or (state[0+i,3] == ith)
        elif corner == (3,0):
            return (state[3-i,0] == ith) or (state[3,i] == ith)
        else:
            return (state[3-i,3] == ith) or (state[3,3-i] == ith)

    def ordered_heuristic(self, state, last_state):
        flat = state.flatten()
        flat.sort()
        flat = np.flip(flat, 0)
        score = 0
        corners = [(0,0), (0,3), (3,0), (3,3)]
        corner_vals = [state[0,0], state[0,3], state[3,0], state[3,3]]
        corner = None
        if flat[0] in corner_vals:
            for i in range(len(corner_vals)):
                if flat[0] == corner_vals[i]:
                    corner = corners[i]
        if corner:
            for j in range(1,3):
                jth = flat[j]
                for i in range(1,4):
                    if self.ith_from_corner(state, corner, i, jth):
                        # score += (2.0 - 0.2 * i) * jth
                        score += jth
        return score

    def empty_tiles_heuristic(self, state, last_state):
        # return float(np.count_nonzero(last_state)) / np.count_nonzero(state)
        return np.count_nonzero(last_state) - np.count_nonzero(state)

    def utility(self, state, last_state):
        h1 = self.corner_heuristic(state, last_state)
        h2 = self.ordered_heuristic(state, last_state)
        h3 = self.empty_tiles_heuristic(state, last_state)
        h4 = 0#1.5 * np.max(state)
        # print(h1,'|',h2,'|',h3,'|',h4)
        return h1 + h2 + h3 + h4
        # return 0.7 * np.sum(state) + self.corner_heuristic(state, last_state) + 2 * self.empty_tiles_heuristic(state, last_state) + self.ordered_heuristic(state, last_state)

    def execute_action(self, board, action):
        board = deepcopy(board)
        if action == 'up':
            board.up()
        elif action == 'down':
            board.down()
        elif action == 'left':
            board.left()
        else:
            board.right()
        return board

    def search(self, board, depth):
        bestVal = 0
        bestAction = 'up'
        for a in self.actions:
            new_board = self.execute_action(board, a)
            if depth == self.max_depth:
                val = self.utility(new_board.grid, board.grid)
            else:
                val = self.utility(new_board.grid, board.grid) + self.discount**(depth+1) * self.search(new_board, depth+1)[1]
            if val > bestVal:
                bestVal = val
                bestAction = a
        return bestAction, bestVal

    def next_action(self, state):
        # import pdb ; pdb.set_trace()
        board = GameBoard()
        board.grid = deepcopy(state)
        self.actions = board.valid_actions()
        action, value = self.search(board, 0)

        return action

