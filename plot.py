import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# show a single frame on a given ax
def generate_frame(frame, ax, args=None):

    if args is not None:
        ax.imshow(frame, origin='lower', cmap='gray', vmin=0, vmax=1, **args)
    else:
        ax.imshow(frame, origin='lower', cmap='gray', vmin=0, vmax=1)

    set_ax(ax)

# show quivers on a given ax
def generate_quiver(vec, ax, args=None):
    height, width, _ = vec.shape

    # generate grid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # flip upside down because the origin of image is at the top left corner
    u = vec[:, :, 0]
    v = vec[:, :, 1]

    if args is not None:
        ax.quiver(X, Y, u, v, **args)
    else:
        ax.quiver(X, Y, u, v)

    set_ax(ax)

# set up ax
def set_ax(ax):
    ax.invert_yaxis()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')


def plot_frame(frames, i, **params):

    filename = params.get('filename', None)

    fig, ax = plt.subplots()

    generate_frame(frames[i, :, :], ax)

    plt.show()

    if filename is not None:
        fig.savefig(filename, dpi=300)


def plot_vector(vec, i, **params):

    frames = params.get('frames', None) # overlay frame if frames is provided
    filename = params.get('filename', None) # save to file if filename is provided
    quiver_args = params.get('quiver_args', None) # arguments for quiver plot
    frame_args = params.get('frame_args', None) # arguments for frame plot

    fig, ax = plt.subplots()

    generate_quiver(vec[i,:,:,:], ax, quiver_args)

    # overlay original frame image when frames data is available
    if frames is not None:
        generate_frame(frames[i, :, :], ax, frame_args)

    plt.show()

    if filename is not None:
        fig.savefig(filename, dpi=300)


def save_vector_video(vec, filename, **params):

    frames = params.get('frames', None)
    quiver_args = params.get('quiver_args', None)
    frame_args = params.get('frame_args', None)

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()

        generate_quiver(vec[i,:,:,:], ax, quiver_args)

        # overlay original frame image when frames data is available
        if frames is not None:
            generate_frame(frames[i, :, :], ax, frame_args)

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=vec.shape[0],
                                  interval=100)
    ani.save(filename, writer='ffmpeg')
    print(f'Video saved to {filename}')