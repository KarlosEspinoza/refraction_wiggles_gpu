from opt_flow import opt_flow
from fluid_flow import fluid_flow
from utils import *
from plot import *
import numpy as np

if __name__ == '__main__':
    # parameters
    video_path = 'data/hand.avi'
    n_jobs = 8  # for parallel computing
    scale = 0.1  # scale factor (0-1) for video frames
    t_filter_theta = 2  # bandwidth of temporal filter
    alpha = 0.04  # weight of the smoothness term for optical flow
    beta2 = 5  # weight of the smoothness term for fluid flow
    beta3 = 1  # weight of the magnitude penalty term for fluid flow
    t_window = 10  # number of frames to be concatenated to form wiggle feature

    # solver parameters
    opt_flow_args = {'atol': 1e-4, 'btol': 1e-4}
    fluid_flow_args = {'atol': 1e-4, 'btol': 1e-4}

    # convert video to frames with shape (n_frames, height, width)
    frames = video2matrix(video_path, scale=scale)

    # apply temporal filter to frames
    frames = temporal_filter(frames, t_filter_theta)
    #frames = frames[:10, :, :]

    #plot_frame(frames, 0, filename='hand.png')

    # compute wiggles (optical flow)
    wiggles, wiggles_var = opt_flow(frames,
                                    alpha2=alpha,
                                    n_jobs=n_jobs,
                                    solver_args=opt_flow_args)

    #plot_vector(wiggles, 0)

    save_vector_video(wiggles, 'hand_wiggles.mp4')

    # compute fluid flow
    flow, flow_var = fluid_flow(wiggles,
                                wiggles_var,
                                beta2=beta2,
                                beta3=beta3,
                                t_window=t_window,
                                n_jobs=n_jobs,
                                solver_args=fluid_flow_args)

    # flow vector field quiver plot
    #plot_vector(flow, 0)

    # flow vector field color plot
    #plot_vector_color(flow,
    #                  0,
    #                  colorwheel=True,
    #                  filename='hand_flow_color.png',
    #                  file_args={'dpi': 300})

    # flow vector field variance plot
    #plot_vector_variance(flow_var,
    #                     0,
    #                     var_max=0.12,
    #                     colorbar=False,
    #                     filename='hand_flow_var.png',
    #                     file_args={'dpi': 300},
    #                     contour_args={
    #                         'cmap': 'plasma',
    #                         'levels': 25
    #                     })

    save_vector_video(flow,
                      'hand_flow.mp4',
                      frames=frames,
                      quiver_args={'color': 'orange'},
                      frame_args={'alpha': 0.7})
