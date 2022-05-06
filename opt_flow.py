import numpy as np
from scipy import signal, sparse
import time
from joblib import Parallel, delayed
from utils import init_parallelization


# compute wiggle features for a single frame
def solve_v(Ix, Iy, It, iframe, **params):
    height, width = Ix.shape

    sqrt_alpha1 = np.sqrt(params['alpha1'])
    sqrt_alpha2 = np.sqrt(params['alpha2'])

    # formulate problem as Tikhonov regularization
    # objective function:
    # min_v alpha1*||(Ix, Iy) * (vx; vy) + It||^2 + alpha2*||L * (vx; vy)||^2
    # where L is Tikhonov matrix
    # solution can be obtained by solving linear system
    #  (sqrt_alpha1*(Ix, Iy); sqrt_alpha2*L) * (vx;  vy) = (-sqrt_alpha1 * It; 0)
    #  |------- A1 --------|  |---- A2 ----|
    #  |---------------- A ----------------|   |-- v --|   |-------- b ---------|

    start_time = time.time()

    # construct sparse matrix A1
    # shape = (height * width, height * width * 2)
    # A1 diagram (height = width = 2):
    #   Ix        Iy
    # x 0 0 0 | x 0 0 0
    # 0 x 0 0 | 0 x 0 0
    # 0 0 x 0 | 0 0 x 0
    # 0 0 0 x | 0 0 0 x

    A1_i = np.repeat(np.arange(height * width), 2)
    A1_j = np.reshape(np.arange(height * width * 2),
                      (2, height * width)).flatten('F')
    A1_values = sqrt_alpha1 * np.vstack(
        (Ix.flatten('F'), Iy.flatten('F'))).flatten('F')
    A1 = sparse.csr_matrix((A1_values, (A1_i, A1_j)),
                           shape=(height * width, height * width * 2))

    # construct sparse matrix A2y
    # shape = (2 * (height-1) * width, height * width * 2)
    # A2y diagram (height = width = 2):
    # -x  x  0  0 |  0  0  0  0
    #  0  0 -x  x |  0  0  0  0
    # -------------------------
    #  0  0  0  0 | -x  x  0  0
    #  0  0  0  0 |  0  0 -x  x

    A2y_i = np.repeat(np.arange((height - 1) * width), 2)
    idx = np.arange(height * width).reshape((height, width),
                                            order='F')[1:, :].flatten('F')
    A2y_j = np.vstack((idx, idx - 1)).flatten('F')
    A2y_values = np.tile([sqrt_alpha2, -sqrt_alpha2], (height - 1) * width)
    A2y = sparse.csr_matrix((A2y_values, (A2y_i, A2y_j)),
                            shape=((height - 1) * width, height * width * 2))
    A2y = sparse.vstack([
        A2y,
        sparse.csr_matrix((A2y_values, (A2y_i, A2y_j + height * width)),
                          shape=((height - 1) * width, height * width * 2))
    ])

    # construct sparse matrix A2x
    # shape = (2 * height * (width-1), height * width * 2)
    # -x  0  x  0 |  0  0  0  0
    #  0 -x  0  x |  0  0  0  0
    # -------------------------
    #  0  0  0  0 | -x  0  x  0
    #  0  0  0  0 |  0 -x  0  x

    A2x_i = np.repeat(np.arange(height * (width - 1)), 2)
    idx = np.arange(height * width)[height:]
    A2x_j = np.vstack((idx, idx - height)).flatten('F')
    A2x_values = np.tile([sqrt_alpha2, -sqrt_alpha2], height * (width - 1))
    A2x = sparse.csr_matrix((A2x_values, (A2x_i, A2x_j)),
                            shape=(height * (width - 1), height * width * 2))
    A2x = sparse.vstack([
        A2x,
        sparse.csr_matrix((A2x_values, (A2x_i, A2x_j + height * width)),
                          shape=(height * (width - 1), height * width * 2))
    ])

    # combine A1, A2x and A2y to obtain A
    A = sparse.vstack([A1, A2x, A2y])
    A = A.astype(np.float32)

    # create vector b
    b = -sqrt_alpha1 * It.flatten('F')
    b = np.concatenate([b, np.zeros(A2x.shape[0] + A2y.shape[0])])
    b = b.astype(np.float32)

    # solve linear system to obtain v
    vi = sparse.linalg.lsmr(A, b)[0]

    # compute variance
    vi_variance = A.T.dot(A) * 1e6

    print(
        f'frame {iframe}/{params["nframe"]-2} is done ({time.time() - start_time:.3f}s)'
    )

    return vi, vi_variance


def opt_flow(frames, **params):
    # default values for parameters
    default_params = {
        'alpha1': 1,
        'alpha2': 1,
        'normalize_It': True,
        'n_jobs': 1  # number of parallelized jobs
    }

    params = {**default_params, **params}

    nframe, height, width = frames.shape
    print(f'nframe: {nframe}, height: {height}, width: {width}')
    params['nframe'] = nframe

    # initialize outputs (for parallelization)
    # v: mean of motion vector, shape = (nframe - 1, height, width, 2)
    # v_varaince: variance of motion vector (sparse matrix for each frame),
    # shape = (nframe - 1, height*width*2, height*width*2)
    delayed_v = []

    # compute dI/dx
    Ix = signal.fftconvolve(frames, [[[1, 0, -1]]], mode='valid')
    Ix = np.pad(Ix, ((0, 0), (0, 0), (1, 1)), 'edge')

    # compute dI/dy
    Iy = signal.fftconvolve(frames, [[[1], [0], [-1]]], mode='valid')
    Iy = np.pad(Iy, ((0, 0), (1, 1), (0, 0)), 'edge')

    # use random values to replace small values
    mask = abs(Ix) < 1e-3
    Ix[mask] = 1e-3 * (np.random.rand(np.count_nonzero(mask)) + 1)
    mask = abs(Iy) < 1e-3
    Iy[mask] = 1e-3 * (np.random.rand(np.count_nonzero(mask)) + 1)

    # compute dI/dt
    It = frames[1:, :, :] - frames[:-1, :, :]

    # normalize dI/dt
    if params['normalize_It']:
        #It = It / np.max(abs(It))
        sd_It = np.sqrt(np.mean(It**2, axis=(1, 2)))
        median_It = np.median(sd_It)
        It = It * median_It / sd_It[:, np.newaxis, np.newaxis]

    n_jobs = init_parallelization(params['n_jobs'])

    # loop through all frames in the video
    print('Start computing wiggle features...')
    for iframe in range(nframe - 1):
        # compute wiggle features

        #vi, vi_variance = solve_v(Ix[iframe, :, :], Iy[iframe, :, :],
        #                          It[iframe, :, :], **params)
        #v[iframe, :, :, :] = vi.reshape(height, width, 2, order='F')
        #v_variance.append(vi_variance)

        # parallelization
        delayed_solve = delayed(solve_v)(Ix[iframe, :, :], Iy[iframe, :, :],
                                         It[iframe, :, :], iframe, **params)
        delayed_v.append(delayed_solve)

    # parallelization
    parallel_pool = Parallel(n_jobs=n_jobs)
    res = parallel_pool(delayed_v)

    # convert results (wiggles and variances) to numpy array
    v = np.array([r[0].reshape(height, width, 2, order='F') for r in res])
    v_variance = [r[1] for r in res]

    return v, v_variance