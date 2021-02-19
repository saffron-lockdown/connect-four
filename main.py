from copy import deepcopy
import random

BOARD_SIZE = 7


class Game:

    _width = BOARD_SIZE
    _height = BOARD_SIZE

    def __init__(self, verbose=True):
        self._board = [[] for _ in range(self._width)]
        self._player = 0
        self._winner = None
        self._move_num = 0
        self._move = None
        self._verbose = verbose

    def move(self, col_num):
        """
        Places a counter in the column with index `column_num`
        """

        # Return if game already won
        if self._winner is not None:
            return self._winner, deepcopy(self._board)

        self._move_num += 1
        self._move = col_num

        # Return if board full
        if sum([len(col) for col in self._board]) == self._width * self._height:
            self.print(msg="board full!")
            return -1, deepcopy(self._board)

        # Return if move is invalid
        if col_num < 0 or col_num >= self._width:
            self.print(msg="move not recognised!")
            self._player = 1 - self._player
            return self._winner, deepcopy(self._board)

        # Return if more than 2*BOARD_SIZE^2 moves played
        if self._move_num > 2 * self._width * self._height:
            self.print(msg="move limit reached")
            return -1, deepcopy(self._board)

        col = self._board[col_num]

        # Return if no space for counter
        if len(col) >= self._height:
            self.print(msg="no space for counter!")
            self._player = 1 - self._player
            return self._winner, deepcopy(self._board)

        col.append(self._player)

        # Check win condition
        if self.winner():
            self._winner = self._player
            self.print(msg=f"WINNER: PLAYER {str(self._winner)}!")
        else:
            self.print()
            self._player = 1 - self._player

        return self._winner, deepcopy(self._board)

    def check_line(self, line):
        """
        Takes a list of lists representing coordinates on the board. E.g.
        [[col1, row1], [col2, row2]]
        Returns True if all counters at those positions belong to the same player
        """
        try:
            result = [self._board[j][i] for [i, j] in line]
        except:
            return False
        return len(result) == 4 and len(set(result)) == 1

    def winner(self):
        """
        Returns True if the game has a horizontal, vertical or diagonal line of four counters
        belonging to the same player
        """
        horizontal_seeds = [
            [i, j] for i in range(0, self._width - 3) for j in range(0, self._height)
        ]
        vertical_seeds = [[j, i] for [i, j] in horizontal_seeds]
        up_diag_seeds = [
            [i, j]
            for i in range(0, self._width - 3)
            for j in range(0, self._height - 3)
        ]
        down_diag_seeds = [
            [i, j] for i in range(0, self._width - 3) for j in range(3, self._height)
        ]

        for [i, j] in horizontal_seeds:
            if self.check_line([[i + k, j] for k in range(4)]):
                return True

        for [i, j] in vertical_seeds:
            if self.check_line([[i, j + k] for k in range(4)]):
                return True

        for [i, j] in up_diag_seeds:
            if self.check_line([[i + k, j + k] for k in range(4)]):
                return True

        for [i, j] in down_diag_seeds:
            if self.check_line([[i + k, j - k] for k in range(4)]):
                return True

        return False

    def print(self, msg=None):
        """
        Prints a command line representation of the board
        """

        if not self._verbose:
            return

        # Header
        print(f"MOVE: {str(self._move_num)}")
        print(f"PLAYER {str(self._player)} PLAYS {str(self._move)}")
        print("-" * 8)

        # Board
        copy = deepcopy(self._board)

        for col in copy:
            col += [" "] * max(0, self._height - len(col))
        rows = list(zip(*copy))

        for row in reversed(rows):
            print("".join([str(item) for item in row]))

        # Footer
        if msg:
            print("-" * 8)
            print(msg)

        print("=" * 8)


####################
# Game Loop
####################

# To make a move, pass the column number to Game.move().
# The function returns the winning player and the updated board


def run_game(alg0, alg1, verbose=0):
    winner = None
    g = Game(verbose=verbose)
    board = g._board

    while True:
        move = alg0.move(board, as_player=0)
        winner, board = g.move(move)

        if winner is not None:
            return winner

        move = alg1.move(board, as_player=1)
        winner, board = g.move(move)

        if winner is not None:
            return winner


def play_game(opponent):
    winner = None
    g = Game(verbose=True)
    board = g._board

    while True:
        move = opponent.move(board, as_player=0)
        winner, board = g.move(move)

        if winner is not None:
            return winner

        print("".join([str(x) for x in range(BOARD_SIZE)]))

        move = None
        while move is None:
            choice = int(input(f"choose column 0-{BOARD_SIZE-1}: "))
            if choice in range(BOARD_SIZE):
                move = choice

        winner, board = g.move(move)

        if winner is not None:
            return winner


####################
# Player Strategies
####################


def alg0(board):
    # always goes for col 2
    return 2


def alg1(board):
    # plays randomly
    return random.randint(0, 7)


"""
winner = run_game(alg0, alg1, verbose=True)
print(winner)
"""
