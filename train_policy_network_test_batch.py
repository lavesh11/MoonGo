from network_helpers import create_network, load_network, get_stochastic_network_move, save_network
from moon_toe import random_player, flat_move_to_tuple, emptyboard, Agent, available_moves, apply_move, gameover, \
    printboard
import tensorflow as tf
import os,collections
import numpy as np
import random
import math



def train_policy_gradient(network_file_path,save_network_file_path=None,learn_rate=1e-3,number_of_games=50000,print_results_every=1000,batch_size=100):

    print 'parameters => LR : ',learn_rate,' Batch Size : ',batch_size
    save_network_file_path = save_network_file_path or network_file_path
    actual_move_placeholder = tf.placeholder("float", shape=(None, 100))
    input_layer, output_layer, variables = create_network(100,(100,100,100),output_softmax=False)

    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=actual_move_placeholder))
    #error = tf.reduce_sum(tf.square(tf.subtract(actual_move_placeholder, output_layer)), reduction_indices=1)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(error)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)

        mini_batch_board_states, mini_batch_moves = [], []

        def my_player(board_state, side):
            #printboard(board_state)
            mini_batch_board_states.append(np.ravel(board_state) * side)
            a1 = Agent(side, lossval=-1)
            move_tuple = a1.random_greedy(board_state)
            move = np.zeros(100)
            move[move_tuple[0]*10+move_tuple[1]] = 1.
            mini_batch_moves.append(move)
            return move_tuple

        def make_training_move(board_state, side):
            a1 = Agent(side, lossval=-1)
            move = a1.action(board_state)
            return move


        game_length = 0
        #count = set()
        for episode_number in range(1, number_of_games):
            #print 'episode no ',episode_number
            if bool(random.getrandbits(1)):
                #print 'network goes first'
                board_state = emptyboard()
                player_turn = 1
                while True:
                    _available_moves = list(available_moves(board_state))

                    if len(_available_moves) == 0:
                        break
                    if player_turn > 0:
                        #move = random_player(board_state, 1)
                        move = make_training_move(board_state, 1)
                        #_ = my_player(board_state, 1)
                        #print 'network move position ', move
                    else:
                        move = my_player(board_state, -1)
                        #print 'player move position ', move

                    if move not in _available_moves:
                        print 'illegal move'
                        break

                    board_state = apply_move(board_state, move, player_turn)
                    #print board_state

                    winner = gameover(board_state)
                    if winner != 0 and winner != 2:
                        break
                    player_turn = -player_turn
                #printboard(board_state)
                #count.add(tuple(np.array(board_state).ravel()))
            else:
                #print 'player goes first'
                board_state = emptyboard()
                player_turn = -1
                while True:
                    _available_moves = list(available_moves(board_state))

                    if len(_available_moves) == 0:
                        break
                    if player_turn > 0:
                        #move = random_player(board_state, 1)
                        move = make_training_move(board_state, 1)
                        #_ = my_player(board_state, 1)
                        #print 'network move position ', move
                    else:
                        move = my_player(board_state, -1)
                        #print 'player move position ', move

                    if move not in _available_moves:
                        print 'illegal move'
                        break


                    board_state = apply_move(board_state, move, player_turn)
                    #print board_state

                    winner = gameover(board_state)
                    if winner != 0 and winner != 2:
                        break
                    player_turn = -player_turn
                #printboard(board_state)
                #count.add(tuple(np.array(board_state).ravel()))

            last_game_length = len(mini_batch_board_states) - game_length
            game_length += last_game_length

            if episode_number % batch_size == 0:
                np_mini_batch_board_states = np.array(mini_batch_board_states).reshape(game_length,*input_layer.get_shape().as_list()[1:])

                ol, _ = session.run([output_layer, train_step], feed_dict={input_layer: np_mini_batch_board_states,
                                                                           actual_move_placeholder: mini_batch_moves})

                # print np.array(np_mini_batch_board_states).reshape(10,10)
                # print 'output_layer_move', np.argmax(ol)
                # print 'our_moves', np.argmax(mini_batch_moves)
                #print np.array(ol).shape,np.array(mini_batch_moves).shape
                correct = np.sum(np.argmax(ol,axis=1) == np.argmax(mini_batch_moves,axis=1))
                del mini_batch_board_states[:]
                del mini_batch_moves[:]

                print episode_number, ': ', 'accuracy ', correct / float(game_length)
                #print 'distinct final states ', len(count)

                game_length = 0

            if episode_number % print_results_every == 0:
                if network_file_path:
                    save_network(session, variables, save_network_file_path)

        if network_file_path:
            print 'saving final network'
            save_network(session, variables, save_network_file_path)

    return variables

# MoonGo_supervised_EE1 quadratic cost, lr = 1e-2, batchsize=10
# MoonGo_supervised_EE_3 quadratic cost, lr = 1e-3, batchsize=0
# MoonGo_supervised_EE quadratic cost, lr = 1e-3, batchsize=0
# MoonGo_supervised_cross cross entropy cost, lr = 1e-3, batchsize=100

if __name__ == '__main__':
    train_policy_gradient("MoonGo_supervised_cross.pickle",save_network_file_path="MoonGo_supervised_cross.pickle",learn_rate=1e-3,number_of_games=1000000,batch_size=100,print_results_every=5000)