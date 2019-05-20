import itertools

import utils


def mover_iter():
    g.start_game()
    yield None
    for player in itertools.islice(itertools.cycle(g.players), 2000):
        best_move = player.find_best_move()
        if best_move:
            g.plot.title_artist.set_text(f'{player.name} to move')
            piece, to, arrow = best_move
            print(f'found best move?! '
                  f'{utils.print_pos(player.pieces[piece].i, player.pieces[piece].j)} - '
                  f'{utils.print_pos(*to)} - '
                  f'{utils.print_pos(*arrow)}')
            g.make_move(player.i_player, piece, *to, *arrow)
        else:
            print(f'Player {player.name} loses!')
            g.plot.title_artist.set_text(f'{player.name} loses!')
            break
        yield None
    while True:
        yield None


if __name__ == "__main__":
    with_plot = True
    g = utils.Game('A4D1G1J4', (10, 10), with_plot=with_plot)

    mover = mover_iter()

    def init_plot():
        pieces_artists = (item.artist for p in g.players for item in p.pieces)
        arrow_artists = (arrow.artist for arrow in g.arrows)
        text_artists = (g.plot.title_artist, )
        return tuple(itertools.chain(pieces_artists, arrow_artists, text_artists))

    def update(_):
        next(mover)
        pieces_artists = (item.artist for p in g.players for item in p.pieces)
        arrow_artists = (arrow.artist for arrow in g.arrows)
        text_artists = (g.plot.title_artist, )
        return tuple(itertools.chain(pieces_artists, arrow_artists, text_artists))

    if with_plot:
        from matplotlib import pyplot as plt
        from matplotlib import animation

        ani = animation.FuncAnimation(g.plot.fig, update, frames=100, interval=100,
                                      init_func=init_plot, blit=True)
        ani.save('animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
        plt.show()
    else:
        for _ in mover:
            g.board.print()
