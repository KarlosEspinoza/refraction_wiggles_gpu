import numpy as np
import cv2
from scipy import signal, interpolate
import warnings


# convert video to image matrix
# shape = (# of frames, height, width)
def video2matrix(video_path, grayscale=True, normalize=True, scale=1.0):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_array = []
    for i in range(frame_count):
        ret, frame = cap.read()

        # scale
        frame = cv2.resize(
            frame, (int(frame_width * scale), int(frame_height * scale)))

        # convert frame to grayscale
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if normalize:
            frame = frame / 255.0

        if ret:
            frame_array.append(frame)
        else:
            break

    cap.release()
    return np.array(frame_array)


# generate a rotationally symmetric Gaussian lowpass filter
# similar to MATLAB's fspecial('gaussian', shape, sigma)
def gaussian_filter(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def temporal_filter(frames, t_filter_theta):
    tw = t_filter_theta * 2
    md_frame = t_filter_theta * 3

    # filter
    t_filter = gaussian_filter((tw * 2 + 1, 1), t_filter_theta)
    t_filter = t_filter[:, np.newaxis]

    frames = signal.fftconvolve(frames, t_filter, mode='same')

    return frames[md_frame:-md_frame, :, :]


# use scipy.interpolate.interp2d to interpolate and evaluate as pairs of values
def interp2d_pairs(*args, **kwargs):
    # internal function that evaluates pairs of values
    def interpolant(x, y, f):
        x, y = np.asarray(x), np.asarray(y)

        return (interpolate.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2],
                                            f.tck[3], f.tck[4], x.ravel(),
                                            y.ravel())[0]).reshape(x.shape)

    return lambda x, y: interpolant(x, y, interpolate.interp2d(
        *args, **kwargs))


def interp2d_pairs_eval(x, y, v, xq, yq):
    f = interp2d_pairs(x, y, v)

    # suppress runtime warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(xq, yq)
