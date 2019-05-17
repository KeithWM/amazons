import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
from typing import List


class Occupier(Enum):
    EMPTY = 0
    ARROW = 1
    PIECE_W = 2
    PIECE_B = 3


class Item:
    radii = {Occupier.ARROW: 0.1, Occupier.PIECE_W: 0.3, Occupier.PIECE_B: 0.3}
    colours = {Occupier.ARROW: 'r', Occupier.PIECE_W: 'w', Occupier.PIECE_B: 'k'}

    def __init__(self, occ: Occupier, i: int, j: int, artist):
        self.occ = occ
        self.i = i
        self.j = j
        self.radius = self.radii[occ]
        self.colour = self.colours[occ]
        self.artist = artist

    @classmethod
    def wo_artist(cls, occ: Occupier, i=-1, j=-1):
        return cls(occ, i, j, None)

    def move(self, i_to, j_to):
        print(f'Moving artist for to {i_to, j_to}')
        self.artist.center = (i_to, j_to)
        self.i = i_to
        self.j = j_to


class Board:
    def __init__(self, m, n, array):
        self.m = m
        self.n = n
        self.array = array

    @classmethod
    def empty(cls, board_dims):
        if isinstance(board_dims, tuple):
            assert len(board_dims) == 2, f'Invalid board dimensions {board_dims}'
            m, n = board_dims
        else:
            m, n = board_dims, board_dims

        return cls(m, n, np.zeros((m, n), dtype=np.uint8))

    def updated(self, array):
        return __class__(self.m, self.n, array)

    def add_obj(self, i, j, occ: Occupier):
        assert (i < self.m) and (j < self.n), f'({i}, {j}) out of range for board size ({self.m}, {self.n})'
        assert self.array[i, j] == Occupier.EMPTY.value, f'Square not empty, ' \
                                                         f'but occupied by {Occupier(self.array[i,j]).name}'

        array = self.array.copy()
        array[i, j] = occ.value
        return self.updated(array)

    def make_move(self, i_from, j_from, i_to, j_to, i_arrow, j_arrow, occ: Occupier):
        assert self.array[i_from, j_from] == occ.value, f'From square not occupied by piece of type {occ.name}, ' \
                                                        f'but occupied by {Occupier(self.array[i_from, j_from]).name}'

        array = self.array.copy()
        # move piece
        assert self._check_valid(array, i_from, j_from, i_to, j_to), \
            'Not a valid path between start and end position'
        array[i_from, j_from], array[i_to, j_to] = array[i_to, j_to], array[i_from, j_from]

        # fire arrow
        assert self._check_valid(array, i_to, j_to, i_arrow, j_arrow), 'Not a valid path for arrow'
        array[i_arrow, j_arrow] = Occupier.ARROW.value
        return self.updated(array)

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
        if (d_i == 0) and (d_j == 0):
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


class Player:
    def __init__(self, n_pieces, i_player):
        assert i_player in (0, 1), f'More than two players not yet supported'
        self.occ = Occupier.PIECE_W if i_player == 0 else Occupier.PIECE_B
        self.pieces = [Item.wo_artist(self.occ) for _ in range(n_pieces)]


class Plot:
    def __init__(self, board, players: List[Player], arrows: List[Item]):
        m, n = board.m, board.n
        checker = (np.arange(m, dtype=int)[:, None] - np.arange(n, dtype=int)[None, :]) % 2

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(checker, cmap='gray', interpolation='nearest', vmin=-2, vmax=2)

        for player in players:
            for piece in player.pieces:
                self.add_artist(piece)
        for arrows in arrows:
            self.add_artist(arrows)

    def add_artist(self, piece):
        print(f'Adding artist for {piece.occ} at {piece.i, piece.j}')
        artist = plt.Circle((piece.i, piece.j), piece.radius, color=piece.colour)
        self.ax.add_artist(artist)
        piece.artist = artist


class Game:
    def __init__(self, n_pieces, board_dims, n_players=2):
        self.n_pieces = n_pieces
        self.n_players = n_players

        self.board = Board.empty(board_dims)
        self.players = [Player(n_pieces, i) for i in range(n_players)]
        self.arrows = [Item.wo_artist(Occupier.ARROW) for _ in range(int(np.prod(board_dims)))]
        self.n_arrows = 0
        self.plot = Plot(self.board, self.players, self.arrows)

    def start_game(self):
        j = 0
        for i, piece in enumerate(piece for player in self.players for piece in player.pieces):
            piece.move(i, j)
            self.board = self.board.add_obj(i, j, piece.occ)

    def make_move(self, i_player, i_piece, i_to, j_to, i_arrow, j_arrow):
        piece = self.players[i_player].pieces[i_piece]
        try:
            self.board =  self.board.make_move(piece.i, piece.j, i_to, j_to, i_arrow, j_arrow, piece.occ)
        except AssertionError as e:
            print(e)
            raise AssertionError
        else:
            piece.move(i_to, j_to)
            self.arrows[self.n_arrows].move(i_arrow, j_arrow)
            self.n_arrows += 1
