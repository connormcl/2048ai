from gamelogic import *
from util import *
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import tensorflow as tf

class RandomAgent(object):
    """An agent that selects its moves randomly"""
    def __init__(self):
        self.actions = [0, 1, 2, 3]

    def next_action(self, state):
        return np.random.choice(self.actions)

class HeuristicAgent(object):
    """An agent that searches for the best move according to its heuristic functions"""
    def __init__(self):
        self.actions = [0, 1, 2, 3]
        self.max_depth = 3
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
        if action == 0:
            board.up()
        elif action == 1:
            board.down()
        elif action == 2:
            board.left()
        else:
            board.right()
        return board

    def search(self, board, depth):
        bestVal = 0
        bestAction = 0
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

class QLearningAgent(object):
    """An agent that learns by Reinforcement Learning"""
    def __init__(self, lr=0.1):
        self.input = tf.placeholder(shape=[None,256], dtype=tf.float32)
        self.W1 = tf.Variable(tf.random_uniform([256,100],0,0.1))
        self.W2 = tf.Variable(tf.random_uniform([100,200],0,0.1))
        self.W3 = tf.Variable(tf.random_uniform([200,4],0,0.1))
        self.h1 = tf.nn.sigmoid(tf.matmul(self.input,self.W1)) # 1x100
        self.h2 = tf.nn.sigmoid(tf.matmul(self.h1,self.W2)) # 1x200
        self.Qout = tf.nn.sigmoid(tf.matmul(self.h2,self.W3))# 1x4
        self.predict = tf.argmax(self.Qout,1)
        self.lr = lr

    def generate_training_data(self, model_agent, num_games=10):
        X = []
        y = []
        print('Generating training data...')
        for i in range(num_games):
            board = GameBoard()
            commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}
            largest_tile = 2
            while not board.is_game_over():
                X.append(convert_state(board.grid))
                a = model_agent.next_action(board.grid)
                y.append(a)

                commands[a]()
                board.add_tile()
                if board.largest_tile > largest_tile:
                    largest_tile = board.largest_tile
                    print('\tGot', largest_tile, 'tile')
            print('(%s/%s) games complete' % (i+1,num_games))
        X, y = np.array(X), np.array(y)
        print('Saving training data...')
        with open('training_data.pkl','wb') as output:
            pickle.dump(X, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(y, output, pickle.HIGHEST_PROTOCOL)
        print('Done.')
        return X, y

    def load_training_data(self):
        print('Loading training data...')
        with open('training_data.pkl','rb') as data:
            X = pickle.load(data)
            y = pickle.load(data)
        print('Done.')
        return X, y

    def one_hot_labels(self, y):
        one_hot = np.zeros((y.size, y.max()+1))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def pretrain(self, model_agent, load_data=True, restore_model=True):
        if load_data:
            X, y = self.load_training_data()
        else:
            X, y = self.generate_training_data(model_agent)

        y = self.one_hot_labels(y)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        y = tf.placeholder("float", shape=[None, y.shape[1]])
        
        # import pdb ; pdb.set_trace()
        self.pretrain_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.Qout))
        l1_regularizer = tf.contrib.layers.l1_regularizer(
           scale=0.00005, scope=None
        )
        weights = tf.trainable_variables() # all vars of the graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

        self.pretrain_loss_op = self.pretrain_loss_op + regularization_penalty
        self.pretrain_update_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.pretrain_loss_op)

        sess = tf.Session()
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        if restore_model:
            print('Restoring model...')
            saver.restore(sess, "/tmp/2048model.ckpt")
            print('Done.')
        else:
            sess.run(init)

        eval_every = 5
        eval_i = 0
        for epoch in range(100):
            for i in range(len(X_train)):
                sess.run(self.pretrain_update_op, feed_dict={self.input: X_train[i: i+1], y: y_train[i: i+1]})

            if eval_i % eval_every == 0:
                train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(self.predict, feed_dict={self.input: X_train, y: y_train}))
                test_accuracy = np.mean(np.argmax(y_test, axis=1) == sess.run(self.predict, feed_dict={self.input: X_test, y: y_test}))

                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                      % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
            eval_i += 1

        save_path = saver.save(sess, "/tmp/2048model.ckpt")
        print("Model saved in path: %s" % save_path)
        import os
        os.system('say "pretraining complete"')

        sess.close()

    def train(self, restore_model=True):
        sess = tf.Session()
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        if restore_model:
            print('Restoring model...')
            saver.restore(sess, "/tmp/2048model2.ckpt")
            print('Done.')
        else:
            sess.run(init)

        nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
        self.train_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=nextQ, logits=self.Qout))
        self.train_update_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.train_loss_op)

        y = .99
        epsilon = 0.1
        num_episodes = 100
        replay_buffer = ExperienceReplayBuffer()

        for i in range(num_episodes):
            board = GameBoard()
            actions = range(4)
            commands = {0: board.up, 1: board.down, 2: board.left, 3: board.right}
            largest_tile = 0

            while not board.is_game_over():
                a,allQ = sess.run([self.predict,self.Qout],feed_dict={self.input: convert_state(board.grid).reshape((1,256))})
                a = a[0]
                allQ2 = deepcopy(allQ[0])
                while not board.action_is_valid(a):
                    allQ2[a] = -1
                    a = np.argmax(allQ2)

                if np.random.rand(1) < epsilon:
                    a = np.random.choice(actions)
                    while not board.action_is_valid(a):
                        a = np.random.choice(actions)

                last_board = deepcopy(board)
                commands[a]()
                r = board.score
                board.add_tile()
                if board.largest_tile > largest_tile:
                    largest_tile = board.largest_tile
                    print('\tGot', largest_tile, 'tile')

                e = Experience(last_board, a, r, deepcopy(board))
                replay_buffer.add(e)

                batch = replay_buffer.uniform_sample()

                for e in batch:
                    board_t, a_t, r_t, board_t_plus_one = e.unpack()

                    Q1 = sess.run(self.Qout,feed_dict={self.input: convert_state(board_t_plus_one.grid).reshape((1,256))})

                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    targetQ[0,a_t] = r_t + y*maxQ1

                    _ = sess.run([self.train_update_op],feed_dict={self.input: convert_state(board_t.grid).reshape((1,256)),nextQ:targetQ})

            print('(%s/%s) games complete' % (i+1,num_episodes))

        save_path = saver.save(sess, "/tmp/2048model2.ckpt")
        print("Model saved in path: %s" % save_path)
        import os
        os.system('say "training complete"')

        sess.close()
