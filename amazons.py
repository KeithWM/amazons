import numpy as np
from enum import Enum
from matplotlib import pyplot as plt
from matplotlib import animation


class Player(Enum):
    WHITE = 0
    BLACK = 1


class Occupier(Enum):
    EMPTY = 0
    ARROW = 1
    PIECE_W = 2
    PIECE_B = 3

class Plot:
    def __init__(self, m, n):
        checker = (np.arange(m, dtype=int)[:, None] - np.arange(n, dtype=int)[None, :]) % 2

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(checker, cmap='gray', interpolation='nearest', vmin=-2, vmax=2)


class Item:
    radii = {Occupier.ARROW: 0.1, Occupier.PIECE_W: 0.3, Occupier.PIECE_B: 0.3}
    colours = {Occupier.ARROW: 'r', Occupier.PIECE_W: 'w', Occupier.PIECE_B: 'k'}

    def __init__(self, occ: Occupier, i: int, j: int, artist):
        self.occ = occ
        self.i = i
        self.j = j
        self.artist = artist

    @classmethod
    def new(cls, occ: Occupier, i: int, j: int, plot: Plot):
        artist = plt.Circle((i, j), cls.radii[occ], color=cls.colours[occ])
        plot.ax.add_artist(artist)
        return cls(occ, i, j, artist)


    def move(self, i_to, j_to):
        self.artist.center = (i_to, j_to)
        self.i = i_to
        self.j = j_to



class Board:
    def __init__(self, m: int, n: int,
                 array: np.ndarray, items: list, plot: Plot):
        self.m = m
        self.n = n
        self.array = array
        self.items = items
        self.plot = plot
        self.n_arrows = (array == Occupier.ARROW.value).sum(axis=None)

    @classmethod
    def empty(cls, m: int, n=None):
        n: int = m if n is None else n

        plot = Plot(m, n)
        items = {
            Occupier.PIECE_W: [],
            Occupier.PIECE_B: [],
            Occupier.ARROW: []
        }
        return cls(m, n, np.zeros((m, n), dtype=np.uint8), items, plot)

    def updated(self, new_array, new_items):
        return __class__(self.m, self.n, new_array, new_items, self.plot)

    def place_obj(self, i, j, occ: Occupier):
        assert (i < self.m) and (j < self.n), f'({i}, {j}) out of range for board size ({self.m}, {self.n})'
        assert self.array[i, j] == Occupier.EMPTY.value, f'Square not empty, ' \
                                                         f'but occupied by {Occupier(board.array[i,j]).name}'

        self.array[i, j] = occ.value
        self.items[occ].append(Item.new(occ, i, j, self.plot))

    def make_move(self, i_from, j_from, i_to, j_to, i_arrow, j_arrow, player: Player):
        occ = Occupier.PIECE_W if player == Player.WHITE else Occupier.PIECE_B
        assert self.array[i_from, j_from] == occ.value, f'From square not occupied by piece of player {player.name}, ' \
                                                        f'but occupied by {Occupier(board.array[i_from, j_from]).name}'

        # move piece
        assert self._check_valid(self.array, i_from, j_from, i_to, j_to), 'Not a valid path between start and end position'
        self.array[i_from, j_from], self.array[i_to, j_to] = self.array[i_to, j_to], self.array[i_from, j_from]
        selected_items = [item for item in self.items if item.i == i_from and item.j == j_from and item.occ == occ]
        assert len(selected_items) == 1, f'Found {selected_items} instead of just one item.'
        selected_items[0].move(i_to, j_to)

        # fire arrow
        assert self._check_valid(self.array, i_to, j_to, i_arrow, j_arrow), 'Not a valid path for arrow'
        self.array[i_arrow, j_arrow] = Occupier.ARROW.value
        self.items[Occupier.ARROW].append(Item.new(Occupier.ARROW, i_arrow, j_arrow, self.plot))

    @staticmethod
    def _check_valid(array, i_from, j_from, i_to, j_to):
        d_i = i_to - i_from
        d_j = j_to - j_from
        n_steps = max(abs(d_i), abs(d_j))
        if abs(d_i) not in (0, n_steps):
            print('Incorrect motion for i')
            return False
        if abs(d_j) not in (0, n_steps):
            print('Incorrect motion for j')
            return False
        if ((d_i == 0) and (d_j == 0)):
            print('No move!')
            return False
        d_i //= n_steps
        d_j //= n_steps
        i, j = i_from, j_from
        for i_step in range(n_steps):
            i += d_i
            j += d_j
            if array[i, j] != Occupier.EMPTY.value:
                print(f'Square ({i}, {j}) not empty between ({i_from}, {j_from}) and ({i_to}, {j_to}).')
                return False
        return True


class Decider:
    def __init__(self, player):
        self.player = player

def mover_iter():
    board.make_move(0, 4, 2, 4, 2, 5, Player.WHITE)
    yield None
    board.make_move(7, 3, 2, 3, 2, 2, Player.BLACK)
    yield None


if __name__ == "__main__":
    board = Board.empty(8)
    mover = mover_iter()

    board.place_obj(0, 4, Occupier.PIECE_W)
    board.place_obj(7, 3, Occupier.PIECE_B)

    def init_plot():
        return tuple(item.artist for item in board.items)

    def update(frame):
        next(mover)
        return tuple(item.artist for item in board.items)


    ani = animation.FuncAnimation(board.plot.fig, update, frames=5, interval=1000,
                                  init_func=init_plot, blit=True)
    plt.show()
