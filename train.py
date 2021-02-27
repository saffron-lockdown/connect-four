from main import Game, run_game, BOARD_SIZE, play_game
from copy import deepcopy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tqdm import tqdm
import pandas as pd
import datetime
import random
import numpy as np
import os
import math
import time
from model import Model, BasicModel, Me

# Tensorflow setting
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def training_loop(training_model, opponent_model, verbose=False):

    winner = None

    # for tensor board logging
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(
    #     log_dir=log_dir, histogram_freq=1
    # )

    # now execute the q learning
    y = 0.9
    eps = 0.5
    interval_size = 100
    num_episodes = interval_size * 1
    decay_factor = (1000*eps)**(-1/num_episodes) # ensures that eps = 0.001 after `num_episodes` episodes

    win_rates = []
    r_avg_list = []
    wins = []
    moves_played = [0] * BOARD_SIZE
    for i in tqdm(range(num_episodes), desc="Training"):

        as_player = random.choice([0, 1])
        eps *= decay_factor

        g = Game(verbose=True)

        if as_player == 1:  # Training as player 1 so opponent makes first move
            winner, board = g.move(opponent_model.move(g._board, 0))
        else:
            board = g._board

        done = False
        r_sum = 0
        move_num = 0
        while not done:
            move_num += 1

            # To encourage early exploration
            if np.random.random() < eps:
                move = np.random.randint(0, BOARD_SIZE - 1)
            else:
                move = training_model.move(board, as_player, print_probs=True)
                preds = training_model.predict(board, as_player)
                moves_played[move] += 1

            winner, new_board = g.move(move)

            if winner is None:
                opponent_move = opponent_model.move(new_board, 1 - as_player)
                winner, new_board = g.move(opponent_move)

            # Calculate reward amount
            if winner == as_player:
                done = True
                wins.append(1)
                r = 100
            elif winner == 1 - as_player:
                done = True
                wins.append(0)
                r = -100
            elif winner == -1:
                done = True
                wins.append(None)
                r = 100
            else:
                r = move_num

            print(
                f"""
                {training_model._name} training as player: {as_player}, eps: {round(eps, 2)},
                reward: {r}, average last 20 games: {sum(r_avg_list[-20:])/20}
                """
            )
            target = r + y * np.max(training_model.predict(new_board, as_player))
            target_vec = training_model.predict(board, as_player)[0]
            target_vec[move] = target

            # Do I need to update fit now that I trainn as p1 and p2?
            training_model.fit_one(
                as_player, board, np.array([target_vec]), epochs=1, verbose=0
            )  # , callbacks=[tensorboard_callback])

            new_preds = training_model.predict(board, as_player)
            board = new_board
            r_sum += r

        if verbose and ((i % interval_size == 0 and i > 0) or (i==num_episodes-1)):
            run_game(training_model, opponent_model, verbose=True)
            win_rates.append(
                performance_stats(training_model, BasicModel(), verbose=True)
            )
            print("win_rates:")
            print(win_rates)
        r_avg_list.append(round(r_sum, 2))


def performance_stats(model1, model2, verbose=False, N_RUNS=50):

    wins = loss = draw = 0
    model1._moves = []

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
        f"As player 0: wins/draws/losses = {100*wins/N_RUNS}/{100*draw/N_RUNS}/{100*loss/N_RUNS}% +/={round(100*ci,1)}%"
    )

    print("moves played:")
    print(pd.Series(model1._moves).value_counts(normalize=True).sort_index())

    win_rate = wins
    wins = loss = draw = 0
    model1._moves = []

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
        f"As player 1: wins/draws/losses = {100*wins/N_RUNS}/{100*draw/N_RUNS}/{100*loss/N_RUNS}% +/={round(100*ci,1)}%"
    )
    print("moves played:")
    print(pd.Series(model1._moves).value_counts(normalize=True).sort_index())

    return (1.0 * (win_rate + wins)) / (2 * N_RUNS)


# basic = BasicModel()
# models = {
#     "model1": Model(),
#     "model2": Model(),
#     "model3": Model(),
# }

# for _ in range(5):

#     training_model_key = random.choice(models.keys())
#     training_model = models[training_model_key]

#     opponent_model_key = random.choice(models.keys())
#     opponent_model = models[opponent_model_key]
#     training_loop(training_model, basic, verbose=True)

# Take a models through basic training
m1 = Model(load_model_name='m1.model')
m2 = Model(load_model_name='m2.model')
for i in range(5):
    print(f"\n\n\nround {i}\n\n\n")
    time.sleep(5)
    training_loop(m1, m2, verbose=True)
    training_loop(m2, m1, verbose=True)

    m1.save("m1.model")
    m2.save("m2.model")


# basic = BasicModel()
# #training_loop(training_model, basic, verbose=True)
# training_loop(training_model, Me(), verbose=True)
# training_model.save("from_scratch_against_me_new_reward.model")


# training_model.save("from_scratch_against_basic.model")

# Train against itself
# training_model = Model("from_scratch_against_basic.model")
# opponent_model = Model("from_scratch_against_basic.model")
# training_loop(training_model, opponent_model, verbose=True)
# training_loop(training_model, basic, verbose=True)
# training_model.save("v_itself_w_basic_training.model")

# # Train against itself
# training_model = Model("v_itself_w_basic_training.model")
# opponent_model = Model("v_itself_w_basic_training.model")
# training_loop(training_model, opponent_model, verbose=True)
# training_loop(training_model, basic, verbose=True)
# training_model.save("v_itself_w_basic_training2.model")

# performance_stats(Model("from_scratch_against_basic.model"), BasicModel(), N_RUNS=500)
# performance_stats(Model("v_itself_w_basic_training.model"), BasicModel(), N_RUNS=500)
# performance_stats(Model("v_itself_w_basic_training2.model"), BasicModel(), N_RUNS=500)

# run_game(training_model, basic_model, verbose=True)
play_game(m1)


# training_model = Model(load_model_name="a.model")
