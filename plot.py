import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_frame(frames, i):
    frame = frames[i, :, :]

    fig, ax = plt.subplots()
    ax.imshow(frame, cmap='gray', vmin=0, vmax=1)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    plt.show()


def plot_quiver(vec, i, ax, args):
    _, height, width, _ = vec.shape

    # generate grid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # flip upside down because the origin of image is at the top left corner
    u = vec[i, :, :, 0]
    v = vec[i, :, :, 1]

    if args is not None:
        ax.quiver(X, Y, u, v, **args)
    else:
        ax.quiver(X, Y, u, v)

    #return X, Y, u, v


def plot_vector(vec, i, **params):

    frames = params.get('frames', None)
    quiver_args = params.get('quiver_args', None)

    fig, ax = plt.subplots()

    plot_quiver(vec, i, ax, quiver_args)

    # overlay original frame image when frames data is available
    if frames is not None:
        frame = frames[i, :, :]
        ax.imshow(frame, origin='lower', cmap='gray', vmin=0, vmax=1)

    ax.invert_yaxis()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')

    plt.show()


def save_vector_video(vec, filename, **params):

    frames = params.get('frames', None)
    quiver_args = params.get('quiver_args', None)

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()

        plot_quiver(vec, i, ax, quiver_args)

        # overlay original frame image when frames data is available
        if frames is not None:
            frame = frames[i, :, :]
            ax.imshow(frame, origin='lower', cmap='gray', vmin=0, vmax=1)

        ax.invert_yaxis()
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=vec.shape[0],
                                  interval=100)
    ani.save(filename, writer='ffmpeg')
    print(f'Video saved to {filename}')