from network_helpers import get_stochastic_network_move, load_network
import functools
from moon_toe import emptyboard, available_moves, flat_move_to_tuple,apply_move,gameover
from network_helpers import create_network
import tensorflow as tf

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

if __name__ == '__main__':
    input_layer, output_layer, variables = create_network(100,(100,100,100))

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        # MoonGo_supervised_cross_prob MoonGo_reinforcement
        load_network(session, variables, 'MoonGo_supervised_cross_prob.pickle')
        while 1:
            board_state = emptyboard()
            player_turn = 1

            while True:
                printboard(board_state)
                _available_moves = list(available_moves(board_state))

                if len(_available_moves) == 0:
                    print("no moves left, game ended a draw")
                    break
                if player_turn > 0:
                    action = raw_input('your move? ')
                    move = (int(action.split(',')[0]), int(action.split(',')[1]))
                else:
                    print 'move probabilites'
                    print get_stochastic_network_move(session, input_layer, output_layer, board_state, -1,valid_only=True)
                    print '\n'*3
                    #move  = monte_carlo_tree_search_uct(game_spec,board_state,-1,10000)[1]
                    move = flat_move_to_tuple(get_stochastic_network_move(session, input_layer, output_layer, board_state, -1,valid_only=True).argmax())

                if move not in _available_moves:
                    print("illegal move ", move)
                    break

                board_state = apply_move(board_state, move, player_turn)

                winner = gameover(board_state)
                if winner != 0 and winner != 2:
                    print("we have a winner, side: %s" % player_turn)
                    break
                player_turn = -player_turn
