#
# CS1010FC --- Programming Methodology
#
# Mission N Solutions
#
# Note that written answers are commented out to allow us to run your
# code easily while grading your problem set.
from random import *
import numpy as np
from copy import deepcopy

class GameBoard(object):
    def __init__(self, n=4):
        self.n = n
        self.grid = np.zeros((n,n))
        self.add_tile()
        self.add_tile()
        self.score = 0.0
        self.largest_tile = 2.0

    def add_tile(self, val=2):
        if 0 in self.grid:
            i = np.random.randint(0, self.n)
            j = np.random.randint(0, self.n)
            while(self.grid[i,j] != 0):
                i = np.random.randint(0, self.n)
                j = np.random.randint(0, self.n)
            self.grid[i,j] = val

    def merge(self):
        done = False
        for i in range(self.n):
            for j in range(self.n-1):
                if self.grid[i,j] == self.grid[i,j+1] and self.grid[i,j] != 0:
                    self.grid[i,j] *= 2
                    self.score += self.grid[i,j]
                    if self.grid[i,j] > self.largest_tile:
                        self.largest_tile = self.grid[i,j]
                    self.grid[i,j+1] = 0
                    done = True
        return done

    def slide(self):
        new = np.zeros((self.n,self.n))
        for i in range(self.n):
            count = 0
            for j in range(self.n):
                if self.grid[i,j] != 0:
                    new[i,count] = self.grid[i,j]
                    count += 1
        self.grid = new

    def reverse(self):
        self.grid = np.flip(self.grid, 1)

    def transpose(self):
        self.grid = self.grid.T

    def left(self):
        self.slide()
        self.merge()
        self.slide()
        return self.grid

    def right(self):
        self.reverse()
        self.left()
        self.reverse()
        return self.grid

    def up(self):
        self.transpose()
        self.left()
        self.transpose()
        return self.grid

    def down(self):
        self.transpose()
        self.reverse()
        self.left()
        self.reverse()
        self.transpose()
        return self.grid

    def is_game_over(self):
        if 0 in self.grid:
            return False
        # row merge?:
        for i in range(self.n):
            for j in range(self.n-1):
                if self.grid[i,j] == self.grid[i,j+1] and self.grid[i,j] != 0:
                    return False
        # col merge?:
        for i in range(self.n-1):
            for j in range(self.n):
                if self.grid[i,j] == self.grid[i+1,j] and self.grid[i,j] != 0:
                    return False
        return True

    def valid_actions(self):
        actions = []
        board = deepcopy(self)
        board.up()
        if not np.array_equal(self.grid, board.grid):
            actions.append(0)
        board = deepcopy(self)
        board.down()
        if not np.array_equal(self.grid, board.grid):
            actions.append(1)
        board = deepcopy(self)
        board.left()
        if not np.array_equal(self.grid, board.grid):
            actions.append(2)
        board = deepcopy(self)
        board.right()
        if not np.array_equal(self.grid, board.grid):
            actions.append(3)
        return actions

    def action_is_valid(self, a):
        board = deepcopy(self)
        actions = [board.up, board.down, board.left, board.right]
        actions[int(a)]()

        if np.array_equal(self.grid, board.grid):
            return False
        return True



#######
#Task 1a#
#######

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 1 mark for creating the correct matrix

def new_game(n):
    return GameBoard(n)


###########
# Task 1c #
###########

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 0 marks for completely wrong solutions
# 1 mark for getting only one condition correct
# 2 marks for getting two of the three conditions
# 3 marks for correct checking

def game_state(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j]==2048:
                return 'win'
    for i in range(len(mat)-1): #intentionally reduced to check the row on the right and below
        for j in range(len(mat[0])-1): #more elegant to use exceptions but most likely this will be their solution
            if mat[i][j]==mat[i+1][j] or mat[i][j+1]==mat[i][j]:
                return 'not over'
    for i in range(len(mat)): #check for any zero entries
        for j in range(len(mat[0])):
            if mat[i][j]==0:
                return 'not over'
    for k in range(len(mat)-1): #to check the left/right entries on the last row
        if mat[len(mat)-1][k]==mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1): #check up/down entries on last column
        if mat[j][len(mat)-1]==mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

