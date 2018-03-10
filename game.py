from tkinter import *
from gamelogic import *
from copy import deepcopy

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.grid_cells = []
        self.init_grid()
        self.board = GameBoard()
        self.commands = {   KEY_UP: self.board.up, KEY_DOWN: self.board.down, KEY_LEFT: self.board.left, KEY_RIGHT: self.board.right,
                            KEY_UP_ALT: self.board.up, KEY_DOWN_ALT: self.board.down, KEY_LEFT_ALT: self.board.left, KEY_RIGHT_ALT: self.board.right }
        self.commands2 = {'up': self.board.up, 'down': self.board.down, 'left': self.board.left, 'right': self.board.right}
        self.update_grid_cells()
        
        # self.mainloop()

        # board = GameBoard()
        import time
        # agent = RandomAgent()
        agent = HeuristicAgent()
        print(self.board.grid)
        while not self.board.is_game_over():
            # time.sleep(.1)
            a = agent.next_action(self.board.grid)
            print('action:', a, ' |  score:', self.board.score)
            self.commands2[a]()
            self.board.add_tile()
            print(self.board.grid)
            self.update_grid_cells()
            self.update()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = int(self.board.grid[i,j])
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()
        
    def key_down(self, event):
        key = repr(event.char)
        if key in self.commands:
            self.commands[repr(event.char)]() # execute move
            self.board.add_tile() # add new tile
            self.update_grid_cells()
            print('Score:',self.board.score)

class RandomAgent(object):
    """docstring for RandomAgent"""
    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']

    def next_action(self, state):
        return np.random.choice(self.actions)

class HeuristicAgent(object):
    """docstring for HeuristicAgent"""
    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']
        self.max_depth = 5
        self.discount = 0.9

    def corner_heuristic(self, state):
        highest_tile = np.max(state)
        corners = [state[0,0], state[0,3], state[3,0], state[3,3]]
        if highest_tile in corners:
            return highest_tile * 2
        return 0

    def ith_from_corner(self, state, corner, i, ith):
        if corner == (0,0):
            return (state[(i,0)] == ith) or (state[(0,i)] == ith)
        elif corner == (0,3):
            return (state[(0,3-i)] == ith) or (state[(0+i,3)] == ith)
        elif corner == (3,0):
            return (state[(3-i,0)] == ith) or (state[(3,i)] == ith)
        else:
            return (state[(3-i,3)] == ith) or (state[(3,3-i)] == ith)

    def ordered_heuristic(self, state):
        flat = state.flatten()
        flat.sort()
        score = 0
        corners = [(0,0), (0,3), (3,0), (3,3)]
        corner_vals = [state[0,0], state[0,3], state[3,0], state[3,3]]
        corner = None
        if flat[0] in corner_vals:
            for i in range(len(corner_vals)):
                if flat[0] == corner_vals[i]:
                    corner = corners[i]
        if corner:
            for i in range(1,4):
                ith = flat[i]
                if self.ith_from_corner(state, corner, i, ith):
                    score += 2 * ith
        return score



    def empty_tiles_heuristic(self, state):
        return 16 - np.count_nonzero(state)

    def utility(self, state):
        return np.sum(state) + self.corner_heuristic(state) + self.empty_tiles_heuristic(state) + self.ordered_heuristic(state)

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
        bestAction = None
        for a in self.actions:
            new_board = self.execute_action(board, a)
            if depth == self.max_depth:
                val = self.utility(new_board.grid)
            else:
                val = self.utility(new_board.grid) + self.discount**(depth+1) * self.search(new_board, depth+1)[1]
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
        

        
        

gamegrid = GameGrid()
