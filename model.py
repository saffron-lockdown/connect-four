from main import Game, run_game, BOARD_SIZE
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import random


class BasicModel:
    def move(self, board, as_player):
        # sometimes go for the middle
        if np.random.random() < 0.25:
            return 3
        # sometimes play randomly
        if np.random.random() < 0.25:
            return random.randint(0, BOARD_SIZE - 1)
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


class Model:

    _model = None
    _name = None
    _moves = []

    def __init__(self, load_model_name=None):
        if load_model_name:
            self._model = keras.models.load_model(load_model_name)
        else:
            self.initialise()

    def move(self, board, as_player):

        # Model is trained as player 1
        if as_player == 1:
            input_vector = self.input_encoding(board)
        else:
            reversed_board = [[1 - cell for cell in col] for col in board]
            input_vector = self.input_encoding(reversed_board)
        pred = self._model.predict(input_vector)
        move = np.argmax(pred)
        self._moves.append(move)

        return move

    def predict(self, board):
        return self._model.predict(self.input_encoding(board))

    def initialise(self):
        self._model = Sequential()
        self._model.add(InputLayer(batch_input_shape=(1, 2 * BOARD_SIZE ** 2)))
        self._model.add(Dense(4 * BOARD_SIZE ** 2, activation="sigmoid"))
        self._model.add(Dense(BOARD_SIZE, activation="linear"))
        self._model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    def input_encoding(self, board, length=BOARD_SIZE):
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

    def fit_one(self, board, *args, **kwargs):
        self._model.fit(self.input_encoding(board), *args, **kwargs)

    def save(self, model_name):
        self._model.save(model_name)
