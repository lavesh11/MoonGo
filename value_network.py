from network_helpers import create_network, load_network, get_stochastic_network_move, save_network
from moon_toe import random_player, flat_move_to_tuple, emptyboard, Agent, available_moves, apply_move, gameover, \
    printboard, play_game, state_key
import tensorflow as tf
import os,collections
import numpy as np
from copy import deepcopy
import random
import math
import time

TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000

NUMBER_RANDOM_RANGE = (1, 100)

def train_policy_gradient(network_file_path,save_network_file_path=None,learn_rate=1e-3,number_of_games=50000,print_results_every=1000,batch_size=100):

    print 'parameters => LR : ',learn_rate,' Batch Size : ',batch_size
    save_network_file_path = save_network_file_path or network_file_path
    target_placeholder = tf.placeholder("float", shape=(None, 1))
    input_layer, output_layer, variables = create_network(10,(100,100,100,100,100),output_nodes=1,output_softmax=False)

    error = tf.reduce_sum(tf.square(target_placeholder - output_layer))
    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(error)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)

        def make_training_move(board_state, side):
            a1 = Agent(side, lossval=-1)
            move = a1.action(board_state)
            return move

        def make_move(board_state, side):
            a1 = Agent(side, lossval=-1)
            move = a1.random_greedy(board_state)
            return move


        board_states_training = {}
        board_states_test = []
        episode_number = 0
        board_states_training_input = {}

        while len(board_states_training_input) < TRAIN_SAMPLES + TEST_SAMPLES:
            #if len(board_states_training_input)%100 == 0:
            print 'total games ',len(board_states_training_input)
            board_state = emptyboard()
            current_board_states_test = []
            if bool(random.getrandbits(1)):
                side = 1
            else:
                side = -1
            while True:
                board_state = apply_move(board_state, make_training_move(board_state, side), side)
                current_board_states_test.append(deepcopy(board_state))
                winner = gameover(board_state)
                if winner != 0:
                    if winner == 2:
                        winner = 0
                    break
                side = -side
            for i in range(len(current_board_states_test)):
                board_state_flat = tuple(np.ravel(current_board_states_test[i]))
                # only accept the board_state if not already in the dict
                if board_state_flat not in board_states_training_input:
                    board_states_training[state_key(current_board_states_test[i])] = float(winner)
                    board_states_training_input[board_state_flat] = 1

        # take a random selection from training into a test set
        for _ in range(TEST_SAMPLES):
            sample = random.choice(list(board_states_training.keys()))
            board_states_test.append((sample, board_states_training[sample]))
            del board_states_training[sample]

        board_states_training = list(board_states_training.items())

        test_error = session.run(error, feed_dict={input_layer: [x[0] for x in board_states_test],
                                                   target_placeholder: [[x[1]] for x in board_states_test]})

        while True:
            np.random.shuffle(board_states_training)
            train_error = 0

            for start_index in range(0, len(board_states_training) - batch_size + 1, batch_size):
                mini_batch = board_states_training[start_index:start_index + batch_size]

                batch_error, _ = session.run([error, train_step],
                                             feed_dict={input_layer: [x[0] for x in mini_batch],
                                                        target_placeholder: [[x[1]] for x in mini_batch]})
                train_error += batch_error

            new_test_error = session.run(error, feed_dict={input_layer: [x[0] for x in board_states_test],
                                                           target_placeholder: [[x[1]] for x in board_states_test]})

            print("episode: %s train_error: %s new_test_error: %s test_error: %s" % (episode_number, train_error, new_test_error, test_error))

            if new_test_error > test_error:
                print("train error went up, stopping training")
                break

            test_error = new_test_error
            episode_number += 1

        if network_file_path:
            print 'saving final network'
            save_network(session, variables, save_network_file_path)

    return variables

if __name__ == '__main__':
    train_policy_gradient("MoonGo_reinforcement.pickle",save_network_file_path="MoonGo_reinforcement.pickle",learn_rate=1e-3,number_of_games=100000,batch_size=100,print_results_every=1000)