import random
from copy import deepcopy

BOARD_SIZE = 4
LINE_LENGTH = 3  # the line you need to make


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

        self.horizontal_seeds = [
            [i, j]
            for i in range(0, self._width - LINE_LENGTH + 1)
            for j in range(0, self._height)
        ]
        self.vertical_seeds = [[j, i] for [i, j] in self.horizontal_seeds]
        self.up_diag_seeds = [
            [i, j]
            for i in range(0, self._width - LINE_LENGTH + 1)
            for j in range(0, self._height - LINE_LENGTH + 1)
        ]
        self.down_diag_seeds = [
            [i, j]
            for i in range(0, self._width - LINE_LENGTH + 1)
            for j in range(LINE_LENGTH - 1, self._height)
        ]

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

        # Other player wins if move is invalid
        if col_num < 0 or col_num >= self._width:
            self._winner = 1 - self._player
            self.print(msg=f"move not recognised!\nWINNER: PLAYER {str(self._winner)}!")
            return self._winner, deepcopy(self._board)

        # Return if more than 2*BOARD_SIZE^2 moves played
        if self._move_num > 2 * self._width * self._height:
            self.print(msg="move limit reached")
            return -1, deepcopy(self._board)

        col = self._board[col_num]

        # Other player wins if no space for counter
        if len(col) >= self._height:
            self._winner = 1 - self._player
            self.print(
                msg=f"no space for counter!\nWINNER: PLAYER {str(self._winner)}!"
            )
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
        return len(result) == LINE_LENGTH and len(set(result)) == 1

    def winner(self):
        """
        Returns True if the game has a horizontal, vertical or diagonal line of four counters
        belonging to the same player
        """

        for [i, j] in self.vertical_seeds:
            if self.check_line([[i, j + k] for k in range(LINE_LENGTH)]):
                return True

        for [i, j] in self.horizontal_seeds:
            if self.check_line([[i + k, j] for k in range(LINE_LENGTH)]):
                return True

        for [i, j] in self.up_diag_seeds:
            if self.check_line([[i + k, j + k] for k in range(LINE_LENGTH)]):
                return True

        for [i, j] in self.down_diag_seeds:
            if self.check_line([[i + k, j - k] for k in range(LINE_LENGTH)]):
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
        print("-" * BOARD_SIZE)

        # Board
        copy = deepcopy(self._board)

        for col in copy:
            col += [" "] * max(0, self._height - len(col))
        rows = list(zip(*copy))

        for row in reversed(rows):
            print("".join([str(item) for item in row]))

        # Footer
        if msg:
            print("-" * BOARD_SIZE)
            print(msg)

        print("=" * BOARD_SIZE)


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
        move = alg0.move(board, 0)
        winner, board = g.move(move)

        if winner is not None:
            return winner

        move = alg1.move(board, 1)
        winner, board = g.move(move)

        if winner is not None:
            return winner


def play_game(opponent):
    winner = None
    g = Game(verbose=True)
    board = g._board

    while True:
        move = opponent.move(board, 0)
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
