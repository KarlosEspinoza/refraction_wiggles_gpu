function motionColorVisualizeOverlay(motionVid, vid, weightVid, outColorFile, varargin)

% function motionColorVisualizeOverlay(motionVid, vid, weightVid, outColorFile, varargin)
% 
% Parameters:
%   VideoProfile, frameRate, 

% Set parameters
para.videoProfile = 'Motion JPEG AVI';
para.frameRate = 5;
para.motionList = [];
para.color_scale = [];
para.dcPriorPower = 0.5;
para = propval(varargin, para);

if (length(size(vid)) == 3) && (size(motionVid,4)~=1)
    vid = repmat(permute(vid,[1,2,4,3]),[1,1,3,1]);
end
if ~isa(vid, 'double')
    vid = im2double(vid);
end
if (size(vid,1) ~= size(motionVid,1)) || (size(vid,2) ~= size(motionVid,2))
    vid = resizeVid(vid, [size(motionVid,1),size(motionVid,2)]);
end

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
    colorFrame = im2double(computeColor(motionScaledVid(:,:,1,i), motionScaledVid(:,:,2,i)));
    hsvFrame = rgb2hsv(colorFrame);
    darkChannelWeight = hsvFrame(:,:,2).^para.dcPriorPower;
    if isempty(weightVid)
        weight = darkChannelWeight;
    else
        weight = weightVid(:,:,i).*darkChannelWeight;
    end
    colorFrame = bsxfun(@times, colorFrame, weight) + bsxfun(@times, vid(:,:,:,i), 1-weight);
    colorFrame(colorFrame>1)=1; colorFrame(colorFrame<0)=0;
    writeVideo(writerColorMotion, colorFrame);
end
if ~isempty(outColorFile)
    close(writerColorMotion);
end

end






