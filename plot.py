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


# convect vectors to color image
def generate_flow_color(vec, ax, **params):

    colorwheel = params.get('colorwheel', False)  # show colorwheel
    transparent = params.get('transparent',
                             True)  # use magnitude as transparency
    alpha_min = params.get('alpha_min', 0)  # minimum alpha value

    u = vec[:, :, 0]
    v = vec[:, :, 1]

    mag = np.sqrt(u**2 + v**2)
    angle = np.mod(np.arctan2(v, u), 2 * np.pi)

    # normalize flow direction
    angle /= 2 * np.pi

    # normalize magnitude
    mag = (mag - np.amin(mag)) / (np.amax(mag) - np.amin(mag))

    # colormap
    cmap = middlebury_cmap()

    img = cmap(angle)  # shape = (height, weight, 4)

    if transparent:
        img[:, :,
            3] = mag * (1 -
                        alpha_min) + alpha_min  # use magnitude as alpha value

    ax.imshow(img, origin='lower')

    set_ax(ax)

    # display colorwheel
    # reference: https://github.com/rfezzani/pyimof
    if colorwheel:
        bbox = ax.get_position()
        w, h = bbox.width, bbox.height
        X0, Y0 = bbox.x0, bbox.y0

        x0, y0 = X0 + 0.01 * w, Y0 + 0.79 * h

        fig = ax.get_figure()
        ax2 = fig.add_axes([x0, y0, w * 0.15, h * 0.15], polar=1)
        wheel_rad, wheel_angle = np.mgrid[:1:50j, :2 * np.pi:1025j]
        ax2.pcolormesh(wheel_angle,
                       wheel_rad,
                       wheel_angle,
                       cmap=cmap,
                       shading='auto')

        ax2.set_xticks([])
        ax2.set_yticks([])


# show contour of variance on a given ax
def generate_variance_contour(var, ax, args=None, **params):

    colorbar = params.get('colorbar', True)
    var_max = params.get('var_max', None)

    height, width, _ = var.shape

    # generate grid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # square root of the determinant of the covariance matrix
    var_scalar = np.sqrt(var[:, :, 0] * var[:, :, 2] - var[:, :, 1]**2)

    if var_max is not None:
        var_scalar = np.minimum(var_scalar, var_max)

    # contour plot
    if args is not None:
        cs = ax.contourf(X, Y, var_scalar, **args)
    else:
        cs = ax.contourf(X, Y, var_scalar)

    set_ax(ax)

    # colorbar
    if colorbar:
        fig = ax.get_figure()
        fig.colorbar(cs)


# set up ax
def set_ax(ax):
    ax.invert_yaxis()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')


# save figure
def save_figure(fig, filename, args=None):
    if args is not None:
        fig.savefig(filename, bbox_inches='tight', **args)
    else:
        fig.savefig(filename, bbox_inches='tight')


def plot_frame(frames, i, **params):

    filename = params.get('filename', None)
    file_args = params.get('file_args', None)

    fig, ax = plt.subplots()

    generate_frame(frames[i, :, :], ax)

    plt.show()

    if filename is not None:
        save_figure(fig, filename, file_args)


def plot_vector(vec, i, **params):

    frames = params.get('frames', None)  # overlay frame if frames is provided
    filename = params.get('filename',
                          None)  # save to file if filename is provided
    quiver_args = params.get('quiver_args', None)  # arguments for quiver plot
    frame_args = params.get('frame_args', None)  # arguments for frame plot
    file_args = params.get('file_args', None)  # arguments for file save

    fig, ax = plt.subplots()

    generate_quiver(vec[i, :, :, :], ax, quiver_args)

    # overlay original frame image when frames data is available
    if frames is not None:
        generate_frame(frames[i, :, :], ax, frame_args)

    plt.show()

    if filename is not None:
        save_figure(fig, filename, file_args)


def plot_vector_color(vec, i, **params):
    frames = params.get('frames', None)
    filename = params.get('filename', None)
    frame_args = params.get('frame_args', None)
    file_args = params.get('file_args', None)

    fig, ax = plt.subplots()

    # overlay original frame image when frames data is available
    if frames is not None:
        generate_frame(frames[i, :, :], ax, frame_args)

    generate_flow_color(vec[i, :, :, :], ax, **params)

    plt.show()

    if filename is not None:
        save_figure(fig, filename, file_args)


def plot_vector_variance(var, i, **params):

    contour_args = params.get('contour_args', None)
    filename = params.get('filename', None)
    file_args = params.get('file_args', None)

    fig, ax = plt.subplots()

    generate_variance_contour(var[i, :, :, :], ax, contour_args, **params)

    plt.show()

    if filename is not None:
        save_figure(fig, filename, file_args)


def save_vector_video(vec, filename, **params):

    frames = params.get('frames', None)
    quiver_args = params.get('quiver_args', None)
    frame_args = params.get('frame_args', None)
    scale_factor = params.get('scale_factor', 10)  # for arrow size

    vec_max = np.sqrt(np.sum(np.amax(vec, axis=(0, 1, 2))**2))
    scale = vec_max * scale_factor

    if quiver_args is not None:
        quiver_args['scale'] = scale
    else:
        quiver_args = {'scale': scale}

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()

        generate_quiver(vec[i, :, :, :], ax, quiver_args)

        # overlay original frame image when frames data is available
        if frames is not None:
            generate_frame(frames[i, :, :], ax, frame_args)

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=vec.shape[0],
                                  interval=100)
    ani.save(filename, writer='ffmpeg')
    print(f'Video saved to {filename}')


# This color map is inspried by the color map used in Middlebury's optical flow code. The Python implementation is based on https://github.com/rfezzani/pyimof
def middlebury_cmap():
    col_len = [0, 15, 6, 4, 11, 13, 6]
    col_range = np.cumsum(col_len)
    ncol = col_range[-1]
    cmap = np.zeros((ncol, 3))

    for idx, (i0, i1,
              l) in enumerate(zip(col_range[:-1], col_range[1:], col_len[1:])):
        j0 = (idx // 2) % 3
        j1 = (j0 + 1) % 3
        if idx & 1:
            cmap[i0:i1, j0] = 1 - np.arange(l) / l
            cmap[i0:i1, j1] = 1
        else:
            cmap[i0:i1, j0] = 1
            cmap[i0:i1, j1] = np.arange(l) / l

    return plt.cm.colors.LinearSegmentedColormap.from_list('middlebury',
                                                           cmap).reversed()
