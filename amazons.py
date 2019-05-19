import itertools

import utils


def mover_iter():
    g.start_game()
    yield None
    for player in itertools.islice(itertools.cycle(g.players), 5):
        piece, to, arrow = player.find_best_move()
        print(f'found best move?! {piece.i, piece.j} - {to} - {arrow}')
        g.make_move(player.i_player, piece, *to, *arrow)
        yield None


if __name__ == "__main__":
    with_plot = False
    g = utils.Game(2, 8, with_plot=with_plot)

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

    if with_plot:
        from matplotlib import pyplot as plt
        from matplotlib import animation

        ani = animation.FuncAnimation(g.plot.fig, update, frames=5, interval=100,
                                      init_func=init_plot, blit=True)
        plt.show()
    else:
        for _ in mover:
            g.board.print()
