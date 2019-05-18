from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

import utils

class Scorer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, game, i_player: int):
        pass


class DeltaScorer(Scorer):
    directions = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    BIGINT = 2**6

    def __init__(self):
        super().__init__()

    def __call__(self, game, i_player: int):
        assert i_player in (0, 1), f'More than two players not yet supported'
        us = game.players[i_player]
        them = game.players[1 - i_player]

        our_moves = self.count_squares(game.board, us)
        their_moves = self.count_squares(game.board, them)

        our_squares = np.logical_and(our_moves < their_moves + 2, our_moves != __class__.BIGINT)
        their_squares = np.logical_and(their_moves <= our_moves - 2, their_moves != __class__.BIGINT)
        unreachable_squares = np.logical_and(our_moves == __class__.BIGINT, their_moves == __class__.BIGINT)
        assert our_squares.sum() + their_squares.sum() + unreachable_squares.sum() == np.prod(game.board.array.shape)
        return our_squares.sum(), -their_squares.sum()

    @staticmethod
    def count_squares(board, player):
        counts = np.ones_like(board.array, dtype=np.int8) * __class__.BIGINT
        change = board.array == player.occ.value
        count = 0
        while change.any():
            counts[change] = count
            can_reach = __class__.find_reachable(board.array, counts == count)
            change = np.logical_and(can_reach, counts > count)
            count += 1
        return counts

    @staticmethod
    def find_reachable(array, startings):
        reachable = np.zeros_like(array, dtype=bool)
        for (i_from, j_from) in zip(*startings.nonzero()):
            for (d_i, d_j) in __class__.directions:
                i, j = i_from, j_from
                i += d_i
                j += d_j
                while (0 <= i < array.shape[0] and
                       0 <= j < array.shape[1] and
                       array[i, j] == utils.Occupier.EMPTY.value):
                    reachable[i, j] = True
                    i += d_i
                    j += d_j
        return reachable


if __name__ == "__main__":
    g = utils.Game(2, 4)
    g.start_game()
    g.make_move(0, 0, 1, 1, 1, 2)
    g.make_move(1, 0, 3, 1, 2, 1)

    scorer = DeltaScorer()
    print(scorer(g, 0))
    print(scorer(g, 1))
    # plt.show()

