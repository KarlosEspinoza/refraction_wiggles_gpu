#!/usr/bin/python
from opt_flow import opt_flow
from fluid_flow import fluid_flow
from tools import *
from plot import *
import numpy as np
import sys
#import cv2
import matplotlib.pyplot as plt

video = sys.argv[1]
result = "./result/"
data = "./data/"
video_path = data+video+".avi"

if __name__ == '__main__':
    # parameters
    ## algorithm
    alpha = 0.04  # weight of the smoothness term for optical flow
    ff_beta2 = 30  # weight of the smoothness term for fluid flow
    ff_beta3 = 2e-2  # weight of the magnitude penalty term for fluid flow
    temporal_filter_theta = 2  # bandwidth of temporal filter
    temporal_window_len = 15  # number of frames to be concatenated to form wiggle feature
    wiigle_strngth = 0.002; # avarage magnitude of wiggle feature
    flow_strength = 0.1; # average magnitude of flow
    ## solver parameters
    n_jobs = 8  # for parallel computing
    scale = 0.1  # scale factor (0-1) for video frames
    opt_flow_args = {'atol': 1e-4, 'btol': 1e-4}
    fluid_flow_args = {'atol': 1e-4, 'btol': 1e-4}

    # Read
    frames = video2matrix(video_path, scale=scale) # convert video to frames with shape (n_frames, height, width)

    # apply temporal filter to frames
    frames = temporal_filter(frames, temporal_filter_theta)
    ##save_nparray_video(frames, result+video+"_temporal_filter.mp4", 30)
    #frames = frames[:5, :, :]
    #plot_frame(frames, 0, filename='hand.png')
    #plot_frame(frames, 0)
    #frames = frames[:3, :3, :3]
    #print(frames.shape)
    #print(frames[0])
    #generate_flow_color(wiggles, 0)

    # compute wiggles (optical flow)
    #wiggles, wiggles_var = opt_flow(frames, alpha2=alpha, n_jobs=n_jobs, solver_args=opt_flow_args)
    wiggles, wiggles_var = opt_flow(frames, alpha2=alpha, n_jobs=n_jobs)
    print(wiggles.shape)
    for i in range(40):
        plot_vector_color(wiggles, i, colorwheel=True)

    #save_vector_video(wiggles, result+video+"_wiggles.mp4")
#
#    ## compute fluid flow
#    #flow, flow_var = fluid_flow(wiggles,
#    #                            wiggles_var,
#    #                            beta2=beta2,
#    #                            beta3=beta3,
#    #                            t_window=t_window,
#    #                            n_jobs=n_jobs,
#    #                            solver_args=fluid_flow_args)
#
#    ## flow vector field quiver plot
#    #plot_vector(flow, 0)
#
#    ## flow vector field color plot
#    ##plot_vector_color(flow,
#    ##                  0,
#    ##                  colorwheel=True,
#    ##                  filename='hand_flow_color.png',
#    ##                  file_args={'dpi': 300})
#
#    ### flow vector field variance plot
#    ###plot_vector_variance(flow_var,
#    ###                     0,
#    ###                     var_max=0.12,
#    ###                     colorbar=False,
#    ###                     filename='hand_flow_var.png',
#    ###                     file_args={'dpi': 300},
#    ###                     contour_args={
#    ###                         'cmap': 'plasma',
#    ###                         'levels': 25
#    ###                     })
#
#    #save_vector_video(flow,
#    #                  result+video+"_flow.mp4",
#    #                  frames=frames,
#    #                  quiver_args={'color': 'orange'},
#    #                  frame_args={'alpha': 0.7})
#
