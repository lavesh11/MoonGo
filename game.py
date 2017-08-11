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


if __name__ == '__main__':
    network_file_path = "MoonGo_reinforcement.pickle"
    target_placeholder = tf.placeholder("float", shape=(None, 1))
    input_layer, output_layer, variables = create_network(10, (100, 100, 100, 100, 100), output_nodes=1,
                                                          output_softmax=False)
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            board_state = emptyboard()
            board_state[0][0] = -1
            board_state[0][1] = -1
            board_state[0][2] = -1
            board_state[0][3] = -1
            board_state[0][4] = -1
            load_network(session, variables, network_file_path)
        ol = session.run(output_layer, feed_dict={input_layer: np.array(state_key(board_state)).reshape(1,10)})
        print ol