from utils import gaussian_filter, interp2d_pairs_eval
import numpy as np
from scipy import signal, sparse
import time
from joblib import Parallel, delayed, cpu_count


# compute fluid flow for a single frame
def solve_u(vi, vi_var, A2, A3, iframe, **params):
    frame_list_len, height, width, _ = vi.shape

    sigma_list = np.logspace(np.log10(params['sigma_start']),
                             np.log10(params['sigma_end']),
                             num=params['n_outer_iter'])

    # initialization
    u_mean_t = np.zeros((height, width, 2))

    # objective function:
    # min_u (dvdx*ux + dvdy*uy + dvdt)^T * S * (dvdx*ux + dvdy*uy + dvdt)
    # + beta2*(|dudx|^2 + |dudy|^2) + beta3*|u|^2
    # = min_u (B*u + dvdt)^T * S * (B*u + dvdt) + beta2*||(Dx,Dy)*u||^2 + beta3*|u|^2
    # where S = vi_var, B = (dvdx, dvdy)
    # solution can be obtained by solving the linear system iteratively
    # (B^T * S * B + beta2 * D^2 + beta3 * I) * u = -(B^T * S * dvdt + beta2*D^2*ut + beta3*ut)
    #  |-- BSB --|   |--- A2 --|   |-- A3 -|          |--- BSc ----|   |-- A2 -|
    #  |---------------- A ----------------|          |---------------- b -------------------|

    xdx, ydx = np.meshgrid(np.arange(1, width - 1, dtype=np.float32),
                           np.arange(0, height, dtype=np.float32))
    xdy, ydy = np.meshgrid(np.arange(0, width, dtype=np.float32),
                           np.arange(1, height - 1, dtype=np.float32))
    x_grid, y_grid = np.meshgrid(np.arange(0, width, dtype=np.float32),
                                 np.arange(0, height, dtype=np.float32))

    # loop through sigma list
    for i_out in range(0, params['n_outer_iter']):
        sw = sigma_list[i_out]
        bandwidth = int(np.ceil(sw * 3))
        f = gaussian_filter((bandwidth * 2 + 1, bandwidth * 2 + 1), sw)

        vt = np.pad(vi, ((0, 0), (bandwidth, bandwidth),
                         (bandwidth, bandwidth), (0, 0)), 'edge')
        vt = signal.fftconvolve(vt,
                                f[np.newaxis, :, :, np.newaxis],
                                mode='valid')

        start_time = time.time()
        for i_in in range(0, params['n_inner_iter']):
            x, y = np.meshgrid(np.arange(0, width, dtype=np.float32),
                               np.arange(0, height, dtype=np.float32))
            x += u_mean_t[:, :, 0]
            y += u_mean_t[:, :, 1]
            mask = (x >= 2) & (x <= width - 3) & (y >= 2) & (y <= height - 3)
            xmask = x[mask]
            ymask = y[mask]

            # initialize sparse matrices BSB and BSc
            BSB = sparse.csr_matrix((height * width * 2, height * width * 2), dtype=np.float32)
            BSc = sparse.csr_matrix((height * width * 2, 1), dtype=np.float32)

            fdx = np.array([1, 0, -1])
            for tf in range(frame_list_len - 1):
                dvx_grid = signal.fftconvolve(vt[tf, :, :, :],
                                              fdx[np.newaxis, :, np.newaxis],
                                              mode='valid')
                dvy_grid = signal.fftconvolve(vt[tf, :, :, :],
                                              fdx[:, np.newaxis, np.newaxis],
                                              mode='valid')

                # construct sparse matrix B
                B11_values = np.zeros((height, width), dtype=np.float32)
                B12_values = np.zeros((height, width), dtype=np.float32)
                B21_values = np.zeros((height, width), dtype=np.float32)
                B22_values = np.zeros((height, width), dtype=np.float32)

                # interpolations
                B11_values[mask] = interp2d_pairs_eval(xdx, ydx, dvx_grid[:, :,
                                                                          0],
                                                       xmask, ymask, dtype=np.float32)
                B12_values[mask] = interp2d_pairs_eval(xdy, ydy, dvy_grid[:, :,
                                                                          0],
                                                       xmask, ymask, dtype=np.float32)
                B21_values[mask] = interp2d_pairs_eval(xdx, ydx, dvx_grid[:, :,
                                                                          1],
                                                       xmask, ymask, dtype=np.float32)
                B22_values[mask] = interp2d_pairs_eval(xdy, ydy, dvy_grid[:, :,
                                                                          1],
                                                       xmask, ymask, dtype=np.float32)

                B11 = sparse.csr_matrix((B11_values.flatten('F'), (np.arange(
                    0, height * width), np.arange(0, height * width))),
                                        shape=(height * width, height * width))
                B12 = sparse.csr_matrix((B12_values.flatten('F'), (np.arange(
                    0, height * width), np.arange(0, height * width))),
                                        shape=(height * width, height * width))
                B21 = sparse.csr_matrix((B21_values.flatten('F'), (np.arange(
                    0, height * width), np.arange(0, height * width))),
                                        shape=(height * width, height * width))
                B22 = sparse.csr_matrix((B22_values.flatten('F'), (np.arange(
                    0, height * width), np.arange(0, height * width))),
                                        shape=(height * width, height * width))

                B = sparse.vstack(
                    [sparse.hstack([B11, B12]),
                     sparse.hstack([B21, B22])])
                B = B.astype(np.float32)

                # construct sparse matrix c
                c1 = np.zeros((height, width))
                c2 = np.zeros((height, width))

                # interpolations
                c1[mask] = interp2d_pairs_eval(x_grid, y_grid, vt[tf + 1, :, :,
                                                                  0], xmask,
                                               ymask, dtype=np.float32) - vt[tf, :, :, 0][mask]
                c2[mask] = interp2d_pairs_eval(x_grid, y_grid, vt[tf + 1, :, :,
                                                                  1], xmask,
                                               ymask, dtype=np.float32) - vt[tf, :, :, 1][mask]

                c = sparse.csr_matrix(
                    np.concatenate([c1.flatten('F'),
                                    c2.flatten('F')])[:, np.newaxis])
                c = c.astype(np.float32)

                BA = B.T.dot(vi_var)
                BA = BA.astype(np.float32)

                BSB = BSB + BA.dot(B)
                BSc = BSc + BA.dot(c)

            ui_t = u_mean_t.flatten('F')

            # solve linear system to obtain u
            A = BSB + A2 + A3
            A = A.astype(np.float32)

            b = -(BSc.toarray().flatten() + A2.dot(ui_t) +
                  params['beta3'] * ui_t)
            b = b.astype(np.float32)

            ui = sparse.linalg.lsmr(A, b)[0]

            u_mean_t += ui.reshape((height, width, 2), order='F')
            u_mean_t[u_mean_t > params['u_max']] = params['u_max']
            u_mean_t[u_mean_t < -params['u_max']] = -params['u_max']
            print(
                f'frame {iframe}/{params["nframe"]-2}, i_out = {i_out}/{params["n_outer_iter"]-1}, i_in = {i_in}/{params["n_inner_iter"]-1} is done ({time.time() - start_time:.3f}s)'
            )

    # calculate variance
    B = sparse.csr_matrix((height * width * 2, height * width * 2))
    for tf in range(frame_list_len - 1):
        dvx_grid = np.pad(
            signal.fftconvolve(vt[tf, :, :, :],
                               fdx[np.newaxis, :, np.newaxis],
                               mode='valid'), ((0, 0), (1, 1), (0, 0)), 'edge')
        dvy_grid = np.pad(
            signal.fftconvolve(vt[tf, :, :, :],
                               fdx[:, np.newaxis, np.newaxis],
                               mode='valid'), ((1, 1), (0, 0), (0, 0)), 'edge')
        B11 = sparse.csr_matrix(
            (dvx_grid[:, :, 0].flatten('F'),
             (np.arange(0, height * width), np.arange(0, height * width))),
            shape=(height * width, height * width))
        B12 = sparse.csr_matrix(
            (dvy_grid[:, :, 0].flatten('F'),
             (np.arange(0, height * width), np.arange(0, height * width))),
            shape=(height * width, height * width))
        B21 = sparse.csr_matrix(
            (dvx_grid[:, :, 1].flatten('F'),
             (np.arange(0, height * width), np.arange(0, height * width))),
            shape=(height * width, height * width))
        B22 = sparse.csr_matrix(
            (dvy_grid[:, :, 1].flatten('F'),
             (np.arange(0, height * width), np.arange(0, height * width))),
            shape=(height * width, height * width))
        B += sparse.vstack(
            [sparse.hstack([B11, B12]),
             sparse.hstack([B21, B22])])

    B = B.T.dot(vi_var).dot(B)
    B11 = np.reshape([B[i, i] for i in range(height * width)], (height, width),
                     order='F')
    B12 = np.reshape([B[i, i + height * width] for i in range(height * width)],
                     (height, width),
                     order='F')
    B21 = np.reshape([B[i + height * width, i] for i in range(height * width)],
                     (height, width),
                     order='F')
    B22 = np.reshape([
        B[i + height * width, i + height * width]
        for i in range(height * width)
    ], (height, width),
                     order='F')

    # filter
    sw = params['sigma_var'] * 3
    fs = gaussian_filter([sw * 2 + 1, sw * 2 + 1], params['sigma_var'])
    fs_weight = signal.fftconvolve(np.ones((height, width)), fs, 'same')
    fs = fs / fs[sw, sw]

    B11 = signal.fftconvolve(B11, fs, 'same') / fs_weight
    B12 = signal.fftconvolve(B12, fs, 'same') / fs_weight
    B21 = signal.fftconvolve(B21, fs, 'same') / fs_weight
    B22 = signal.fftconvolve(B22, fs, 'same') / fs_weight
    invdetB = 1 / (B11 * B22 - B12 * B21)
    u_var_t = np.stack((invdetB * B22, -invdetB * B21, invdetB * B11), axis=2)

    return u_mean_t , u_var_t


def fluid_flow(wiggles, wiggles_var, **params):

    # default values for parameters
    default_params = {
        'beta2': 1,
        'beta3': 1e-7,
        'sigma_start': 8,
        'sigma_end': 1,
        'n_outer_iter': 4,
        'n_inner_iter': 1,
        't_window': 2,
        'sigma_var': 3,
        'u_max': 50,
        'n_jobs': 1  # number of parallelized jobs
    }

    params = {**default_params, **params}

    nframe, height, width, _ = wiggles.shape
    params['nframe'] = nframe

    # initialize outputs (for parallelization)
    # u_mean: shape = (nframe - 1, height, width, 2)
    # u_var:  shape = (nframe - 1, height, width, 3)
    delayed_u = []

    # construct sparse matrix Dy
    # shape = ((height-1) * width, height * width)
    # Dy diagram (height = width = 3):
    # -1  1  0  0  0  0  0  0  0
    #  0 -1  1  0  0  0  0  0  0
    #  0  0  0 -1  1  0  0  0  0
    #  0  0  0  0 -1  1  0  0  0
    #  0  0  0  0  0  0 -1  1  0
    #  0  0  0  0  0  0  0 -1  1

    Dy_i = np.repeat(np.arange((height - 1) * width), 2)
    idx = np.arange(height * width).reshape((height, width),
                                            order='F')[1:, :].flatten('F')
    Dy_j = np.vstack((idx, idx - 1)).flatten('F')
    Dy_values = np.tile([1., -1.], (height - 1) * width)
    Dy = sparse.csr_matrix((Dy_values, (Dy_i, Dy_j)),
                           shape=((height - 1) * width, height * width))

    # construct sparse matrix Dx
    # shape = (height * (width - 1), height * width)
    # Dx diagram (height = width = 3):
    # -1  0  0  1  0  0  0  0  0
    #  0 -1  0  0  1  0  0  0  0
    #  0  0 -1  0  0  1  0  0  0
    #  0  0  0 -1  0  0  1  0  0
    #  0  0  0  0 -1  0  0  1  0
    #  0  0  0  0  0 -1  0  0  1

    Dx_i = np.repeat(np.arange(height * (width - 1)), 2)
    idx = np.arange(height * width)[height:]
    Dx_j = np.vstack((idx, idx - height)).flatten('F')
    Dx_values = np.tile([1., -1.], height * (width - 1))
    Dx = sparse.csr_matrix((Dx_values, (Dx_i, Dx_j)),
                           shape=(height * (width - 1), height * width))

    # construct matrix A2
    A2 = params['beta2'] * (Dy.T.dot(Dy) + Dx.T.dot(Dx))
    A2 = sparse.block_diag((A2, A2))

    # construct matrix A3
    A3 = sparse.csr_matrix(
        (params['beta3'] * np.ones(height * width * 2),
         (np.arange(height * width * 2), np.arange(height * width * 2))),
        shape=(height * width * 2, height * width * 2))

    # print out info
    print('Start computing fluid flow...')
    if params['n_jobs'] > 1:
        if params['n_jobs'] > cpu_count():
            print(
                f'Warning: {params["n_jobs"]} jobs requested, but only {cpu_count()} CPUs available.'
            )
            params['n_jobs'] = cpu_count()
        print(f'Running {params["n_jobs"]} parallel jobs...')

    for iframe in range(nframe - 1):

        # extract needed wiggle features for current frame
        iframe_list = np.arange(max(0, iframe - params['t_window']),
                                min(nframe, iframe + 2 + params['t_window']))

        vi = wiggles[iframe_list, :, :, :]
        vi_var = wiggles_var[iframe]

        # u_mean_t, u_var_t = solve_u(vi, vi_var, A2, A3, iframe, **params)

        # parallelization
        delayed_solve = delayed(solve_u)(vi, vi_var, A2, A3, iframe, **params)
        delayed_u.append(delayed_solve)

    # parallelization
    parallel_pool = Parallel(n_jobs=params["n_jobs"])
    res = parallel_pool(delayed_u)

    # motion vectors and variances
    u_mean = np.array([r[0] for r in res])
    u_var = np.array([r[1] for r in res])

    return u_mean, u_var