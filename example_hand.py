from opt_flow import opt_flow
from fluid_flow import fluid_flow
from utils import *
from plot import *

if __name__ == '__main__':
    # parameters
    video_path = 'data/hand.avi'
    n_jobs = 8  # for parallel computing
    scale = 0.05  # scale factor (0-1) for video
    t_filter_theta = 2  # bandwidth of temporal filter
    alpha = 0.04
    beta2 = 5
    beta3 = 1
    t_window = 10  # number of frames to be concatenated to form wiggle feature

    frames = video2matrix(video_path, scale=scale)
    frames = temporal_filter(frames, t_filter_theta)
    #frames = frames[:10, :, :]

    #plot_frame(frames, 0)

    # compute wiggles
    wiggles, wiggles_var = opt_flow(frames, alpha2=alpha)
    plot_vector(wiggles, 0)
    #save_vector_video(wiggles, 'hand_wiggles.mp4')

    # compute fluid flow
    flow, flow_var = fluid_flow(wiggles,
                                wiggles_var,
                                beta2=beta2,
                                beta3=beta3,
                                t_window=t_window,
                                n_jobs=n_jobs)

    plot_vector(flow, 0)
    save_vector_video(flow, 'hand_flow.mp4')