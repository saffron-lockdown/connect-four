from main import Game, run_game, BOARD_SIZE, play_game
from copy import deepcopy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tqdm import tqdm
import pandas as pd

import random
import numpy as np
import os
import math
from model import Model, BasicModel

# Tensorflow setting
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def training_loop(training_model, opponent_model, verbose=False):

    winner = None

    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.90
    interval_size = 5
    num_episodes = interval_size * 10

    r_avg_list = []
    wins = []
    moves_played = [0] * BOARD_SIZE
    for i in tqdm(range(num_episodes), desc="Training"):

        eps *= decay_factor

        g = Game(verbose=False)
        g.move(3)  # Opponent starts by going for the middle
        board = g._board

        done = False
        r_sum = 0
        while not done:

            # To encourage early exploration
            if np.random.random() < eps:
                move = np.random.randint(0, BOARD_SIZE - 1)
            else:
                move = training_model.move(board, as_player=1)
                moves_played[move] += 1

            winner, new_board = g.move(move)

            if winner is None:
                opponent_move = opponent_model.move(new_board, as_player=0)
                winner, new_board = g.move(opponent_move)

            # Calculate reward amount
            if winner == 1:
                done = True
                wins.append(1)
                r = 10
            elif winner == 0:
                done = True
                wins.append(0)
                r = -10
            elif winner == -1:
                done = True
                wins.append(None)
                r = 0.1
            else:
                r = 0

            target = r + y * np.max(training_model.predict(new_board))
            target_vec = training_model.predict(board)[0]
            target_vec[move] = target

            training_model.fit_one(board, np.array([target_vec]), epochs=1, verbose=0)

            board = new_board
            r_sum += r

        if i % interval_size == 0 and verbose:
            print("Episode {} of {}".format(i + 1, num_episodes))
            performance_stats(training_model, opponent_model, verbose=True)
            print(f"Win/Loss/Draw: {nwins}/{nloss}/{ndraw}")
            print(moves_played)
            moves_played = [0] * BOARD_SIZE

        r_avg_list.append(round(r_sum, 2))

    training_model.save("a.model")


def performance_stats(model1, model2, verbose=False):

    wins = loss = draw = 0

    N_RUNS = 50
    for _ in tqdm(range(N_RUNS), desc="Scoring"):
        result = run_game(alg0=model1, alg1=model2, verbose=False)
        if result == 0:
            wins += 1
        elif result == 1:
            loss += 1
        else:
            draw += 1

    p = wins / N_RUNS
    ci = 1.96 * p * (1 - p) / math.sqrt(N_RUNS)
    print(
        f"As player 0: model1/draw/model2 = {100*wins/N_RUNS}/{100*draw/N_RUNS}/{100*loss/N_RUNS}% +/={round(100*ci,1)}%"
    )
    print("last 100 moves:")
    print(pd.Series(model1._moves[-100:]).value_counts())

    wins = loss = draw = 0

    N_RUNS = 50
    for _ in tqdm(range(N_RUNS), desc="Scoring"):
        result = run_game(alg0=model2, alg1=model1, verbose=False)
        if result == 1:
            wins += 1
        elif result == 0:
            loss += 1
        else:
            draw += 1

    p = wins / N_RUNS
    ci = 1.96 * p * (1 - p) / math.sqrt(N_RUNS)
    print(
        f"As player 1: model1/draw/model2 = {100*wins/N_RUNS}/{100*draw/N_RUNS}/{100*loss/N_RUNS}% +/={round(100*ci,1)}%"
    )
    print("last 100 moves:")
    print(pd.Series(model1._moves[-100:]).value_counts())


# Take two models through basic training
training_model = Model(load_model_name="a.model")
opponent_model = training_model
basic_model = BasicModel()


# for i in range(20):
#     print("#"*20)
#     print("Round " + str(i))
#     print("#"*20)
#     training_loop(training_model, opponent_model, verbose=False)
#     performance_stats(training_model, basic_model, verbose=False)

run_game(training_model, basic_model, verbose=True)
play_game(training_model)


# training_model = Model(load_model_name="a.model")
