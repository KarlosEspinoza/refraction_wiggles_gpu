function motionColorVisualize(motionVid, outColorFile, varargin)

% function motionColorVisualize(motionVid, outColorFile, varargin)
% 
% Parameters:
%   VideoProfile, frameRate, 

% Set parameters
para.videoProfile = 'Motion JPEG AVI';
para.frameRate = 5;
para.motionList = [];
para.color_scale = [];
para = propval(varargin, para);

% Write motion
writerColorMotion = VideoWriter(outColorFile, para.videoProfile);
if strcmp(para.videoProfile, 'Motion JPEG AVI') || strcmp(para.videoProfile, 'Motion JPEG 2000') || ...
        strcmp(para.videoProfile, 'MPEG-4')
    writerColorMotion.Quality = 100;
end
writerColorMotion.FrameRate = para.frameRate;
open(writerColorMotion);

% Rescale motion
motionMag = sqrt(sum(motionVid.^2,3));
if isempty(para.color_scale)
    para.color_scale = max(motionMag(:));
end
motionTooLarge = motionMag > para.color_scale;
tscale = zeros(size(motionMag));
tscale(motionTooLarge) = 1 ./ motionMag(motionTooLarge);
tscale(~motionTooLarge) = 1 / para.color_scale;
motionScaledVid = bsxfun(@times, tscale, motionVid);

%% Generate result
for i=1:size(motionScaledVid,4)
    colorFrame = computeColor(motionScaledVid(:,:,1,i), motionScaledVid(:,:,2,i));
    writeVideo(writerColorMotion, colorFrame);
end
if ~isempty(outColorFile)
    close(writerColorMotion);
end

end






