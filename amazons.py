from matplotlib import pyplot as plt
from matplotlib import animation
import itertools

import utils


def mover_iter():
    g.start_game()
    g.make_move(0, 0, 0, 4, 0, 2)
    print('moved')
    yield None
    g.make_move(1, 0, 2, 4, 2, 2)
    print('moved')
    yield None


if __name__ == "__main__":
    g = utils.Game(2, 8)

    mover = mover_iter()

    def init_plot():
        pieces_artists = (item.artist for p in g.players for item in p.pieces)
        arrow_artists = (arrow.artist for arrow in g.arrows)
        return tuple(itertools.chain(pieces_artists, arrow_artists))

    def update(_):
        next(mover)
        pieces_artists = (item.artist for p in g.players for item in p.pieces)
        arrow_artists = (arrow.artist for arrow in g.arrows)
        return tuple(itertools.chain(pieces_artists, arrow_artists))


    ani = animation.FuncAnimation(g.plot.fig, update, frames=5, interval=1000,
                                  init_func=init_plot, blit=True)
    plt.show()
