
from copy import deepcopy
import random

class Game:

    _width = 8
    _height = 8

    def __init__(self):
        self._board = [[] for _ in range(self._width)]
        self._player = 0
        self._winner = None
        self._move = 0

    def move(self, col_num):
        self._move += 1
        
        # Stop if board full
        if sum([len(col) for col in self._board]) == self._width * self._height:
            return -1, self._board

        if col_num < 0 or col_num > self._width:
            print("move not recognised!")
            return self._winner, self._board 
        
        col = self._board[col_num]
        
        # Check space for counter
        if len(col) > self._height:
            print("no space for counter!")
            return self._winner, self._board 
        
        col.append(self._player)

        # Check win condition
        if self.winner():
            self._winner = self._player
        
        self.print()
        self._player = 1 - self._player

        return self._winner, self._board 

    def check_line(self, line):
        try:
            result = [self._board[j][i] for [i, j] in line]
        except:
            return False
        return len(result) == 4 and len(set(result)) == 1
            

    def winner(self):
        
        horizontal_seeds = [[i, j] for i in range(0, self._width - 3) for j in range(0, self._height)]
        vertical_seeds = [[j, i] for [i, j] in horizontal_seeds]
        up_diag_seeds = [[i, j] for i in range(0, self._width - 3) for j in range(0, self._height-3)]
        down_diag_seeds = [[i, j] for i in range(0, self._width - 3) for j in range(3, self._height)]
        
        for [i, j] in horizontal_seeds:
            if self.check_line([[i+k, j] for k in range(4)]):
                return True

        for [i, j] in vertical_seeds:
            if self.check_line([[i, j+k] for k in range(4)]):
                return True

        for [i, j] in up_diag_seeds:
            if self.check_line([[i+k, j+k] for k in range(4)]):
                return True
        
        for [i, j] in down_diag_seeds:
            if self.check_line([[i+k, j-k] for k in range(4)]):
                return True
        

    def print(self):

        copy = deepcopy(self._board)
        
        for col in copy:
            col += [' ']*max(0, self._height-len(col))
        rows = list(zip(*copy))
        
        print("MOVE: "+ str(self._move))
        print("-"*8)
        for row in reversed(rows):
            print("".join([str(item) for item in row]))
        print("="*8)

        if self._winner is not None:
            print("WINNER: PLAYER "+ str(self._winner) + "!")
            print("="*8)


def player1():
    # always goes for col 2
    return 2

def player2():
    # plays randomly
  return random.randint(0, 7)
    
winner = None
g = Game()


# To make a move, pass the column number to Game.move().
# The function returns the winning player and the updated board
while winner is None:

    winner, board = g.move(player1())
    
    if winner is None:
        winner, board = g.move(player2())


