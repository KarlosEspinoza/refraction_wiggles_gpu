addpath(genpath('./tools/'));
addpath(genpath('./alg/'));
inpath = './data/';
outpath = './result/';
%mkdir(outpath);

%% Parameters
vidname = 'hand';
alpha = 0.04;
ffBeta2 = 30;
ffBeta3 = 2e-2;
temporalFilterTheta = 2; % bandwidth of temporal filter
temporalWindowLen = 15;  % No. of frames to be concatenated to form the wiggle feature

% the following parameters are only for visualization
wiggleStrength = 0.002; % Average magnitude of wiggle feature
flowStrength = 0.1;     % Average magnitude of flow

%% Read
vid = vid2mat(fullfile(inpath, [vidname, '.avi']));
vid = im2double(colorvid2gray(vid));

%%% Filter
tw = temporalFilterTheta*2;
mdFrame = temporalFilterTheta*3;
f = fspecial('gaussian', [1,tw*2+1], temporalFilterTheta);
vid = convn(vid, shiftdim(f,-1), 'same');
vid = vid(:,:,mdFrame+1:end-mdFrame);
%vid(2:4, 3:5, 1)
vid = vid(1:3, 1:3, 1:3);

%
%%% Wiggle Feature
[wiggle,wiggleVar] = optflow(vid, 'alpha2', alpha);
%disp(size(wiggle))
disp(wiggle)
%a = wiggleVar(1)
%disp(a{1,1})
motionColorVisualizeOverlay(wiggle, vid.^0.2, [], fullfile(outpath,[vidname,'_wiggle.avi']), 'color_scale', wiggleStrength);
%
%%% Run Fluid Flow
%[flowMean,flowVar] = fluidflow(wiggle, wiggleVar, 'beta2', ffBeta2, 'beta3', ffBeta3, 'timeWin', temporalWindowLen);    
%motionColorVisualizeOverlay(flowMean, repmat(permute(vid.^(1/3),[1,2,4,3]),[1,1,3,1]), [], ...
%    fullfile(outpath,[vidname,'_airflow.avi']),'color_scale',flowStrength);
%motionQuiverVisualize(vid/2, flowMean, fullfile(outpath,[vidname,'_airflow_quiver.avi']), 'quiver_scale', flowStrength);
%
%
%
%
%
%
