import collections
import pickle
import random
import math
from operator import itemgetter
import itertools

import numpy as np

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
DRAW = 2

BOARD_SIZE = 10
LINE_SIZE = 5

NAMES = {0: '_', 1: 'X', -1: 'O'}

def printboard(state):
    cells = []
    print ' ',
    for i in range(BOARD_SIZE):
        print '{0}'.format(str(i).center(5)),
    print '\n'
    for i in range(BOARD_SIZE):
        print i,
        for j in range(BOARD_SIZE):
            print '{0}'.format(NAMES[state[i][j]].center(5)),
        print('\n')

def emptyboard():
    state = []
    for i in range(BOARD_SIZE):
        state.append([])
        for j in range(BOARD_SIZE):
            state[i].append(0)
    return state


def inside(i, j):
    return BOARD_SIZE > i >= 0 and BOARD_SIZE > j >= 0


def gameover(state):
    # Horizontal winning
    total_zero = 0
    for i in range(BOARD_SIZE):
        ct = {0:0, 1:0, -1:0}
        for j in range(LINE_SIZE-1):
            ct[state[i][j]] += 1
            if state[i][j] == 0:
                total_zero+=1
        for j in range(LINE_SIZE-1,BOARD_SIZE):
            if state[i][j] == 0:
                total_zero+=1
            ct[state[i][j]]+=1
            # print i,j,state[i][j],ct[state[i][j]]
            if state[i][j] != 0 and ct[state[i][j]] == LINE_SIZE:
                return state[i][j]
            ct[state[i][j-LINE_SIZE+1]]-=1

    # Vertical winning
    for j in range(BOARD_SIZE):
        ct = {0:0, 1:0, -1:0}
        for i in range(LINE_SIZE-1):
            ct[state[i][j]]+=1
        for i in range(LINE_SIZE-1,BOARD_SIZE):
            ct[state[i][j]]+=1
            if state[i][j] != 0 and ct[state[i][j]] == LINE_SIZE:
                return state[i][j]
            ct[state[i-LINE_SIZE+1][j]]-=1

    # Diagonal winning
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            ct1 = {0:0, 1:0, -1:0}
            ct2 = {0: 0, 1: 0, -1: 0}
            for k in range(LINE_SIZE):
                x,y = i+k,j+k
                if inside(x,y):
                    ct1[state[x][y]]+=1
                x,y = i+k,j-k
                if inside(x,y):
                    ct2[state[x][y]]+=1
            if ct1[1] == LINE_SIZE or ct2[1] == LINE_SIZE:
                return 1
            elif ct1[-1] == LINE_SIZE or ct2[-1] == LINE_SIZE:
                return -1

    if total_zero == 0:
        return 2
    return 0

def random_player(board_state, _):
    moves = list(available_moves(board_state))
    return random.choice(moves)

def available_moves(board_state):
    for x, y in itertools.product(range(BOARD_SIZE), range(BOARD_SIZE)):
        if board_state[x][y] == 0:
            yield (x, y)

def flat_move_to_tuple(move_index):
    return int(move_index / BOARD_SIZE), move_index % BOARD_SIZE

def tuple_move_to_flat(tuple_move):
    return tuple_move[0]*10 + tuple_move[1]

def apply_move(board_state, move, side):
    move_x, move_y = move
    board_state[move_x][move_y] = side
    return board_state

def possible(state):
    line = collections.defaultdict(lambda: 0)
    line.clear()
    for i in range(BOARD_SIZE):
        ct = {0: 0, 1: 0, -1: 0}
        for j in range(LINE_SIZE - 1):
            ct[state[i][j]] += 1
        for j in range(LINE_SIZE - 1, BOARD_SIZE):
            ct[state[i][j]] += 1
            line[(ct[1],ct[-1],ct[0])] += 1
            ct[state[i][j - LINE_SIZE + 1]] -= 1

    for j in range(BOARD_SIZE):
        ct = {0: 0, 1: 0, -1: 0}
        for i in range(LINE_SIZE - 1):
            ct[state[i][j]] += 1
        for i in range(LINE_SIZE - 1, BOARD_SIZE):
            ct[state[i][j]] += 1
            line[(ct[1],ct[-1],ct[0])] += 1
            ct[state[i-LINE_SIZE+1][j]] -= 1

    for i in range(BOARD_SIZE - LINE_SIZE + 1):
        ct = {0: 0, 1: 0, -1: 0}
        for k in range(LINE_SIZE - 1):
            ct[state[i + k][k]] += 1
        inc = LINE_SIZE - 1
        while inside(i + inc, inc):
            ct[state[i + inc][inc]] += 1
            line[(ct[1],ct[-1],ct[0])] += 1
            ct[state[i + inc - LINE_SIZE + 1][inc - LINE_SIZE + 1]] -= 1
            inc += 1
        ct = {0: 0, 1: 0, -1: 0}
        if i != 0:
            for k in range(LINE_SIZE - 1):
                ct[state[k][i + k]] += 1
            inc = LINE_SIZE - 1
            while inside(inc, i + inc):
                ct[state[inc][i + inc]] += 1
                line[(ct[1],ct[-1],ct[0])] += 1
                ct[state[inc - LINE_SIZE + 1][i + inc - LINE_SIZE + 1]] -= 1
                inc += 1

    for i in range(LINE_SIZE - 1, BOARD_SIZE):
        ct = {0: 0, 1: 0, -1: 0}
        for k in range(LINE_SIZE - 1):
            ct[state[k][i - k]] += 1
        inc = LINE_SIZE - 1
        while inside(inc, i - inc):
            ct[state[inc][i - inc]] += 1
            line[(ct[1],ct[-1],ct[0])] += 1
            ct[state[inc - LINE_SIZE + 1][i - inc + LINE_SIZE - 1]] -= 1
            inc += 1
        ct = {0: 0, 1: 0, -1: 0}
        if i != BOARD_SIZE - 1:
            for k in range(LINE_SIZE - 1):
                ct[state[BOARD_SIZE - i - 1 + k][BOARD_SIZE - 1 - k]] += 1
            inc = LINE_SIZE - 1
            while inside(BOARD_SIZE - i - 1 + inc, BOARD_SIZE - 1 - inc):
                ct[state[BOARD_SIZE - i - 1 + inc][BOARD_SIZE - 1 - inc]] += 1
                line[(ct[1],ct[-1],ct[0])] += 1
                ct[state[BOARD_SIZE - i - 1 + inc - LINE_SIZE + 1][BOARD_SIZE - 1 - inc + LINE_SIZE - 1]] -= 1
                inc += 1
    return line


def play_game(plus_player_func, minus_player_func, board_state=None):

    board_state = board_state or emptyboard()
    player_turn = 1

    while True:
        _available_moves = list(available_moves(board_state))

        if len(_available_moves) == 0:
            return 0
        if player_turn > 0:
            move = plus_player_func(board_state, 1)
        else:
            move = minus_player_func(board_state, -1)

        if move not in _available_moves:
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)

        winner = gameover(board_state)
        if winner != 0 and winner != 2:
            return winner
        player_turn = -player_turn

def state_key(state):
    c = possible(state)
    key = (c[(1, 0, 4)], c[(2, 0, 3)], c[(3, 0, 2)], c[(4, 0, 1)], c[(5, 0, 0)], c[(0, 1, 4)], c[(0, 2, 3)],
           c[(0, 3, 2)], c[(0, 4, 1)], c[(0, 5, 0)])
    return key

class Agent(object):
    def __init__(self, player, lossval = 0):
        self.values = {}
        self.player = player
        self.epsilon = 0.1
        self.lossval = lossval
        self.prevstate = None
        self.prevvalue = 0
        self.alpha = 0.99

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == DRAW:
            return 0
        elif winner == EMPTY:
            return 0.5
        else:
            return self.lossval

    def available_moves(self,state):
        for i, j in itertools.product(range(BOARD_SIZE), range(BOARD_SIZE)):
            if state[i][j] == 0:
                yield (i, j)


    def state_formula(self, state):
        c = self.possible(state)
        if self.player == 1:
            key = (c[(1,0,4)],c[(2,0,3)],c[(3,0,2)],c[(4,0,1)],c[(5,0,0)],c[(0,1,4)],c[(0,2,3)],c[(0,3,2)],c[(0,4,1)],c[(0,5,0)])
        else:
            key = (c[(0,1,4)],c[(0,2,3)],c[(0,3,2)],c[(0,4,1)],c[(0,5,0)], c[(1,0,4)],c[(2,0,3)],c[(3,0,2)],c[(4,0,1)],c[(5,0,0)])
        return math.log(key[0]+1)+math.sqrt(key[1])+key[2]+(2*key[3])**2+((5*key[4])**3)-(math.log(key[5]+1)+math.sqrt(key[6])+3*key[7]+(4*key[8])**2)

    def random_greedy(self,state):
        maxval = -999999999
        maxpos = None
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[i][j] == 0:
                    state[i][j] = self.player
                    val = self.state_formula(state)
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxpos = (i, j)
        return maxpos

    def action(self,state):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[i][j] == 0:
                    state[i][j] = self.player
                    val = self.state_formula(state)
                    state[i][j] = EMPTY
                    positions.append((val,(i,j)))
        positions = sorted(positions, key=lambda tup: tup[0], reverse=True)
        #random_index = random.randrange(0, min(2,len(positions)))
        if len(positions) >= 3:
            # initially trained on p=[0.65, 0.25, 0.1] , p=[0.8, 0.15, 0.05]
            random_index = numpy.random.choice(numpy.arange(0, 3), p=[0.8, 0.15, 0.05])
        else:
            random_index = random.randrange(0, min(3, len(positions)))
        return positions[random_index][1]

    def possible(self, state):
        line = collections.defaultdict(lambda: 0)
        line.clear()
        for i in range(BOARD_SIZE):
            ct = {0: 0, 1: 0, -1: 0}
            for j in range(LINE_SIZE - 1):
                ct[state[i][j]] += 1
            for j in range(LINE_SIZE - 1, BOARD_SIZE):
                ct[state[i][j]] += 1
                line[(ct[1],ct[-1],ct[0])] += 1
                ct[state[i][j - LINE_SIZE + 1]] -= 1

        for j in range(BOARD_SIZE):
            ct = {0: 0, 1: 0, -1: 0}
            for i in range(LINE_SIZE - 1):
                ct[state[i][j]] += 1
            for i in range(LINE_SIZE - 1, BOARD_SIZE):
                ct[state[i][j]] += 1
                line[(ct[1],ct[-1],ct[0])] += 1
                ct[state[i-LINE_SIZE+1][j]] -= 1

        for i in range(BOARD_SIZE - LINE_SIZE + 1):
            ct = {0: 0, 1: 0, -1: 0}
            for k in range(LINE_SIZE - 1):
                ct[state[i + k][k]] += 1
            inc = LINE_SIZE - 1
            while inside(i + inc, inc):
                ct[state[i + inc][inc]] += 1
                line[(ct[1],ct[-1],ct[0])] += 1
                ct[state[i + inc - LINE_SIZE + 1][inc - LINE_SIZE + 1]] -= 1
                inc += 1
            ct = {0: 0, 1: 0, -1: 0}
            if i != 0:
                for k in range(LINE_SIZE - 1):
                    ct[state[k][i + k]] += 1
                inc = LINE_SIZE - 1
                while inside(inc, i + inc):
                    ct[state[inc][i + inc]] += 1
                    line[(ct[1],ct[-1],ct[0])] += 1
                    ct[state[inc - LINE_SIZE + 1][i + inc - LINE_SIZE + 1]] -= 1
                    inc += 1

        for i in range(LINE_SIZE - 1, BOARD_SIZE):
            ct = {0: 0, 1: 0, -1: 0}
            for k in range(LINE_SIZE - 1):
                ct[state[k][i - k]] += 1
            inc = LINE_SIZE - 1
            while inside(inc, i - inc):
                ct[state[inc][i - inc]] += 1
                line[(ct[1],ct[-1],ct[0])] += 1
                ct[state[inc - LINE_SIZE + 1][i - inc + LINE_SIZE - 1]] -= 1
                inc += 1
            ct = {0: 0, 1: 0, -1: 0}
            if i != BOARD_SIZE - 1:
                for k in range(LINE_SIZE - 1):
                    ct[state[BOARD_SIZE - i - 1 + k][BOARD_SIZE - 1 - k]] += 1
                inc = LINE_SIZE - 1
                while inside(BOARD_SIZE - i - 1 + inc, BOARD_SIZE - 1 - inc):
                    ct[state[BOARD_SIZE - i - 1 + inc][BOARD_SIZE - 1 - inc]] += 1
                    line[(ct[1],ct[-1],ct[0])] += 1
                    ct[state[BOARD_SIZE - i - 1 + inc - LINE_SIZE + 1][BOARD_SIZE - 1 - inc + LINE_SIZE - 1]] -= 1
                    inc += 1
        return line

def play(agent1, agent2):
    state = emptyboard()
    for k in range(BOARD_SIZE*BOARD_SIZE):
        if k % 2 == 0:
            move = agent1.action(state)
            state[move[0]][move[1]] = agent1.player
        else:
            move = agent2.action(state)
            state[move[0]][move[1]] = agent2.player
        winner = gameover(state)
        if winner != EMPTY:
            # print winner
            return winner

    # print winner
    return winner

class Human(object):
    def __init__(self, player):
        self.player = player

    def action(self, state):
        printboard(state)
        action = raw_input('your move? ')
        move = (int(action.split(',')[0]),int(action.split(',')[1]))
        '''
        while not move in emptystate:
            action = raw_input('your move? ')
            move = (int(action.split(',')[0]), int(action.split(',')[1]))
        '''
        return move

import numpy

if __name__ == '__main__':
    for episode in range(30):
        p = [0.6, 0.1, 0.3]
        move = np.random.multinomial(1, p)
        print episode, move
    '''
    count = set()
    for episode in range(30):
        if bool(random.getrandbits(1)):
            board_state = emptyboard()
            side = 1
            while True:
                _available_moves = list(available_moves(board_state))

                if len(_available_moves) == 0:
                    break
                if side > 0:
                    a1 = Agent(1, lossval=-1)
                    move = a1.action(board_state)
                else:
                    a1 = Agent(-1, lossval=-1)
                    move = a1.random_greedy(board_state)
                    # print 'player move position ', move

                if move not in _available_moves:
                    print 'illegal move'
                    break

                board_state = apply_move(board_state, move, side)
                # print board_state

                winner = gameover(board_state)
                if winner != 0 and winner != 2:
                    break
                side = -side
            count.add(tuple(np.array(board_state).ravel()))
            printboard(board_state)
        else:
            # print 'player goes first'
            board_state = emptyboard()
            side = -1
            while True:
                _available_moves = list(available_moves(board_state))

                if len(_available_moves) == 0:
                    break
                if side > 0:
                    a1 = Agent(1, lossval=-1)
                    move = a1.action(board_state)
                else:
                    a1 = Agent(-1, lossval=-1)
                    move = a1.random_greedy(board_state)

                if move not in _available_moves:
                    print 'illegal move'
                    break

                board_state = apply_move(board_state, move, side)
                # print board_state

                winner = gameover(board_state)
                if winner != 0 and winner != 2:
                    break
                side = -side
            count.add(tuple(np.array(board_state).ravel()))
            printboard(board_state)
        if episode%100 == 0:
            print episode, len(count)
    
    '''