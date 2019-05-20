from abc import ABC, abstractmethod
import numpy as np

import utils


class Scorer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, array, us, them):
        # return a tuple of comparable numbers
        # the higher the numbers the better the performance
        # the numbers are compared in order, the first having the highest impact
        pass


class DeltaScorer(Scorer):
    BIGINT = 2**6

    def __init__(self):
        super().__init__()

    def __call__(self, array: np.ndarray, us, them):

        our_moves = self.count_squares(array, us)
        their_moves = self.count_squares(array, them)

        our_squares = np.logical_and(our_moves < their_moves + 1, our_moves != __class__.BIGINT)
        their_squares = np.logical_and(their_moves <= our_moves - 1, their_moves != __class__.BIGINT)
        unreachable_squares = np.logical_and(our_moves == __class__.BIGINT, their_moves == __class__.BIGINT)
        assert our_squares.sum() + their_squares.sum() + unreachable_squares.sum() == np.prod(array.shape)
        return our_squares.sum() - their_squares.sum(), unreachable_squares.sum()

    @staticmethod
    def count_squares(array, occ):
        counts = np.ones_like(array, dtype=np.int8) * __class__.BIGINT
        change = array == occ.value
        count = 0
        while change.any():
            counts[change] = count
            can_reach = __class__.find_reachable(array, counts == count)
            change = np.logical_and(can_reach, counts > count)
            count += 1
        return counts

    @staticmethod
    def find_reachable(array, startings):
        reachable = np.zeros_like(array, dtype=bool)
        for (i_from, j_from) in zip(*startings.nonzero()):
            for (d_i, d_j) in utils.DIRECTIONS:
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
    from matplotlib import pyplot as plt

    g = utils.Game(1, 4)
    g.start_game()

    scorer = DeltaScorer()
    g.make_move(0, 0, 1, 2, 2, 2)
    print(scorer(g.board.array, g.players[0].us, g.players[0].them))
    print(scorer(g.board.array, g.players[1].us, g.players[1].them))
    plt.show()
