import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


def plot_frame(frames, i):
    frame = frames[i, :, :] * 255
    frame = frame.astype(np.uint8)
    im = Image.fromarray(frame, 'L')
    im.show()


def generate_quiver(vec, i):
    _, height, width, _ = vec.shape

    # generate grid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # flip upside down because the origin of image is at the top left corner
    u = vec[i, ::-1, :, 0]
    v = vec[i, ::-1, :, 1]

    return X, Y, u, v


def plot_vector(vec, i):

    X, Y, u, v = generate_quiver(vec, i)

    fig, ax = plt.subplots()
    ax.quiver(X, Y, u, v)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')

    plt.show()


def save_vector_video(vec, filename):

    fig, ax = plt.subplots()

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')

    def animate(i):
        ax.clear()
        X, Y, u, v = generate_quiver(vec, i)
        ax.quiver(X, Y, u, v)

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=vec.shape[0],
                                  interval=100)
    ani.save(filename, writer='ffmpeg')