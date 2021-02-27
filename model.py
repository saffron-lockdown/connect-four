from main import Game, run_game, BOARD_SIZE
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import random

from scipy.special import softmax


class BasicModel:

    _name = "basic_model"

    def move(self, board, as_player):
        # sometimes go for the middle
        if np.random.random() < 0.25 and len(board[3]) < BOARD_SIZE:
            return 3
        # sometimes play randomly
        if np.random.random() < 0.25:
            rand = random.randint(0, BOARD_SIZE - 1)
            if len(board[rand]) < BOARD_SIZE:
                return rand
        # sometimes cover the first 0 you see
        else:
            for i in range(len(board)):
                if (
                    board[i]
                    and len(board[i]) < BOARD_SIZE
                    and board[i].pop() == 1 - as_player
                ):
                    return i
        return random.randint(0, BOARD_SIZE - 1)

    def fit(self, *args, **kwargs):
        pass


class Me:
    def move(self, board, as_player):
        return int(input(f"choose column 0-{BOARD_SIZE-1}: "))


class Model:

    _model = None
    _name = None
    _moves = []

    def __init__(self, load_model_name=None, model_name='model'):
        if load_model_name:
            self._model = keras.models.load_model(load_model_name)
            self._name = load_model_name
        else:
            self.initialise()
            self._name = model_name

    def move(self, board, as_player, print_probs=False):

        pred = self.predict(board, as_player)

        if print_probs:
            print([round(x, 2) for x in pred[0]])

        smax = softmax(pred[0])
        move = random.choices(range(len(smax)), smax)[0]
        self._moves.append(move)

        return move

    def predict(self, board, as_player):
        return self._model.predict(self.input_encoding(board, as_player))

    def initialise(self):
        self._model = Sequential()
        self._model.add(InputLayer(batch_input_shape=(1, 2 * BOARD_SIZE ** 2)))
        self._model.add(Dense(4 * BOARD_SIZE ** 2, activation="relu"))
        self._model.add(Dense(BOARD_SIZE, activation="linear"))
        self._model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    def input_encoding(self, board, as_player):

        if as_player == 1:
            input_vector = self.board_to_vec(board)
        else:
            reversed_board = [[1 - cell for cell in col] for col in board]
            input_vector = self.board_to_vec(reversed_board)
        return input_vector

    def board_to_vec(self, board, length=BOARD_SIZE):
        copy = deepcopy(board)
        for b in copy:
            b += [None] * (length - len(b))

        input_vec = []

        for col in copy:
            for item in col:
                if item is None:
                    input_vec += [0, 0]
                else:
                    if item == 0:
                        input_vec += [0, 1]
                    elif item == 1:
                        input_vec += [1, 0]
                    else:
                        raise Exception

        return np.array([input_vec])

    def fit_one(self, as_player, board, *args, **kwargs):
        self._model.fit(self.input_encoding(board, as_player), *args, **kwargs)

    def save(self, model_name):
        self._model.save(model_name)
