import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_frame(frames, i):
    frame = frames[i, :, :] * 255
    frame = frame.astype(np.uint8)
    im = Image.fromarray(frame, 'L')
    im.show()


def plot_vector(vec, i):
    _, height, width, _ = vec.shape

    # generate grid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # flip upside down because the origin of image is at the top left corner
    u = vec[i, ::-1, :, 0]
    v = vec[i, ::-1, :, 1]

    fig, ax = plt.subplots()
    ax.quiver(X, Y, u, v)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')

    plt.show()