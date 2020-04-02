import numpy as np


# Util Functions
def random_action():
    """ This function returns a random integer corresponding in range [0,5)"""
    moves =  np.random.random(size=2)*2 - 1
    return moves


def one_hot(moves):
    """ This function converts a 1d array to a one-hot encoding"""
    return [1 if i == maximum(moves) else 0 for i in range(len(moves))]


def maximum(moves):
    """ This function gets the arg max of an array"""
    return np.argmax(moves)


def process_output(moves):
    """ This function converts a neural network output to range [0,1] and then gives the corresponding move"""
    return moves

