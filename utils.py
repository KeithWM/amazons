import re
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
from typing import List

import scorer


DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def print_pos(i, j):
    return chr(ord('A') + j) + str(i + 1)


def unprint_pos(letter, number):
    return int(number) - 1, ord(letter) - ord('A')


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
        self.artist.center = (j_to, i_to)
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

    def print(self):
        d = {Occupier.EMPTY.value: '_',
             Occupier.ARROW.value: 'x',
             Occupier.PIECE_W.value: 'W',
             Occupier.PIECE_B.value: 'B'}
        print('\n'.join(''.join(d[i] for i in row) for row in self.array))


class Player:
    def __init__(self, n_pieces, i_player, game):
        assert i_player in (0, 1), f'More than two players not yet supported'
        self.i_player = i_player
        if i_player == 0:
            self.us = Occupier.PIECE_W
            self.them = Occupier.PIECE_B
            self.name = 'White'
        elif i_player == 1:
            self.us = Occupier.PIECE_B
            self.them = Occupier.PIECE_W
            self.name = 'Black'
        else:
            raise ValueError
        self.pieces = [Item.wo_artist(self.us) for _ in range(n_pieces)]
        self.game = game
        self.board_scorer = scorer.SimpleScorer()

    def find_best_move(self):
        scores = self._score_all_possible_moves()
        if not scores:
            return None
        best_key, best_score = [(key, score) for key, score in scores.items()][0]
        for key, score in scores.items():
            if self._is_better_score(score, best_score):
                best_key, best_score = key, score
        return best_key

    @staticmethod
    def _is_better_score(curr, best):
        for c, b in zip(curr, best):
            if c < b:
                return True
            if c > b:
                return False
            if c == b:
                continue
            else:
                raise ValueError

    def _score_all_possible_moves(self):
        scores = {}
        for i_piece, piece in enumerate(self.pieces):
            for i, j in move_while_empty(self.game.board.array, piece.i, piece.j):
                hypo_array = self.game.board.array.copy()
                hypo_array[piece.i, piece.j], hypo_array[i, j] = hypo_array[i, j], hypo_array[piece.i, piece.j]
                for i_arrow, j_arrow in move_while_empty(hypo_array, i, j):
                    hypo_arrow_array = hypo_array.copy()
                    hypo_arrow_array[i_arrow, j_arrow] = Occupier.ARROW.value
                    # we score the position of the OPPONENT after our own move
                    scores[i_piece, (i, j), (i_arrow, j_arrow)] = self.board_scorer(hypo_arrow_array,
                                                                                    self.them, self.us)
        return scores


def move_while_empty(array: np.ndarray, i_from: int, j_from: int):
    for d_i, d_j in DIRECTIONS:
        i, j = i_from, j_from
        i += d_i
        j += d_j
        while (0 <= i < array.shape[0] and
               0 <= j < array.shape[1] and
               array[i, j] == Occupier.EMPTY.value):
            yield i, j
            i += d_i
            j += d_j


class Plot:
    def __init__(self, board, players: List[Player], arrows: List[Item]):
        m, n = board.m, board.n
        checker = (np.arange(m, dtype=int)[:, None] - np.arange(n, dtype=int)[None, :]) % 2

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(checker, cmap='gray', interpolation='nearest', vmin=-2, vmax=2, origin='lower')

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for i in range(m):
            self.ax.text(-1, i, str(i+1))
        for j in range(n):
            self.ax.text(j, -1, chr(ord('A') + j))

        self.title_artist = self.ax.text(n / 2 - .5, m - .25, 'Title', horizontalalignment='center')

        for player in players:
            for piece in player.pieces:
                self.add_artist(piece)
        for arrows in arrows:
            self.add_artist(arrows)

    def add_artist(self, piece):
        artist = plt.Circle((piece.i, piece.j), piece.radius, color=piece.colour)
        self.ax.add_artist(artist)
        piece.artist = artist


class DummyArtist:
    def __init__(self):
        self.center = (-1, -1)


class DummyPlot:
    def __init__(self, _, players: List[Player], arrows: List[Item]):
        for player in players:
            for piece in player.pieces:
                self.add_artist(piece)
        for arrows in arrows:
            self.add_artist(arrows)

    @staticmethod
    def add_artist(piece):
        piece.artist = DummyArtist()


class Game:
    def __init__(self, position_string: str, board_dims, n_players=2, with_plot=True, symmetry='mirror'):
        self.starting_positions = self._extract_stating_positions(position_string)
        self.n_pieces = len(self.starting_positions)
        self.n_players = n_players

        self.board = Board.empty(board_dims)
        self.players = [Player(self.n_pieces, i, self) for i in range(n_players)]
        self.arrows = [Item.wo_artist(Occupier.ARROW) for _ in np.ndindex(self.board.array.shape)]
        self.n_arrows = 0
        self.plot = Plot(self.board, self.players, self.arrows) if with_plot else \
            DummyPlot(self.board, self.players, self.arrows)
        self.symmetry = symmetry

    @staticmethod
    def _extract_stating_positions(pos_str):
        parts = re.split('([A-Z])', pos_str)
        return [unprint_pos(letter, number) for letter, number in zip(parts[1::2], parts[2::2])]

    def start_game(self):
        # j = 0
        # for i, piece in enumerate(piece for player in self.players for piece in player.pieces):
        #     piece.move(i, j)
        #     self.board = self.board.add_obj(i, j, piece.occ)
        for (i, j), piece_w, piece_b in zip(self.starting_positions, *(player.pieces for player in self.players)):
            piece_w.move(i, j)
            self.board = self.board.add_obj(i, j, piece_w.occ)
            if self.symmetry == 'radial':
                piece_b.move(j, self.board.n - i - 1)
                self.board = self.board.add_obj(j, self.board.n - i - 1, piece_b.occ)
            elif self.symmetry == 'mirror':
                piece_b.move(self.board.m - i - 1, j)
                self.board = self.board.add_obj(self.board.m - i - 1, j, piece_b.occ)
            else:
                raise ValueError

    def make_move(self, i_player, i_piece, i_to, j_to, i_arrow, j_arrow):
        piece = self.players[i_player].pieces[i_piece]
        try:
            self.board = self.board.make_move(piece.i, piece.j, i_to, j_to, i_arrow, j_arrow, piece.occ)
        except AssertionError as e:
            print(e)
            raise AssertionError
        else:
            piece.move(i_to, j_to)
            self.arrows[self.n_arrows].move(i_arrow, j_arrow)
            self.n_arrows += 1
