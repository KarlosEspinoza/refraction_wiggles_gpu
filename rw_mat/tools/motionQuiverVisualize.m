function motionQuiverVisualize(videoOrig, motionVid, outQuiverFile, varargin)

% function motionQuiverVisualize(videoOrig, motionVid, outQuiverFile, varargin)
% 
% Parameters:
%   VideoProfile, frameRate, motionList, quiver_width, quiver_step,
%   quiver_scale, abs_quiver_scale, quiver_headsize, ForceSameFrame,
%   quiver_color

% Set parameters
para.videoProfile = 'Motion JPEG AVI';
para.frameRate = 5;
para.motionList = [];
para.quiver_width = 2;
para.quiver_step = 30;
para.quiver_scale = [];
para.abs_quiver_scale = 0.9;
para.quiver_headsize = 2;
para.ForceSameFrame = false;
para.quiver_color = [1,0.6,0];
para.vidRescale = 1;
para.motionThresh = 0.2;
para = propval(varargin, para);
warning('off', 'images:imshow:magnificationMustBeFitForDockedFigure')

if ~isempty(outQuiverFile) && isempty(videoOrig)
    error('Video Orig cannot be empty when generating quiver video');
end
if length(size(videoOrig)) == 3
    grayScaleVid = true;
else
    grayScaleVid = false;
end
[h,w,~,nF] = size(motionVid);
if para.ForceSameFrame
    if size(motionVid,4) ~= nF
        error('videoOrig and motionVid should be in same scale');
    end
else
    nMotion = size(motionVid,4);
    if isempty(para.motionList)
        step = floor(nF/nMotion);
        motionList = step:step:step*nMotion;
    else
        motionList = para.motionList;
    end
end

% Write motion
writerQuiverMotion = VideoWriter(outQuiverFile, para.videoProfile);
if strcmp(para.videoProfile, 'Motion JPEG AVI') || strcmp(para.videoProfile, 'Motion JPEG 2000') || ...
        strcmp(para.videoProfile, 'MPEG-4')
    writerQuiverMotion.Quality = 100;
end
writerQuiverMotion.FrameRate = para.frameRate;
open(writerQuiverMotion);

% Rescale motion
motionMag = sqrt(sum(motionVid.^2,3));
if isempty(para.quiver_scale)
    para.quiver_scale = max(motionMag(:));
end
motionTooLarge = motionMag > para.quiver_scale;
tscale = zeros(size(motionMag));
tscale(motionTooLarge) = 1 ./ motionMag(motionTooLarge);
tscale(~motionTooLarge) = 1 / para.quiver_scale;
motionScaledVid = bsxfun(@times, tscale, motionVid);
if ~isempty(para.motionThresh)
    motionMag = sum(motionScaledVid.^2,3);
    motionScaledVid(repmat(motionMag<para.motionThresh^2,[1,1,2,1]))=0;
end

%% Generate result
xlist = para.quiver_step:para.quiver_step:w; ylist = para.quiver_step:para.quiver_step:h;
motionIdx = 1;
for i=1:nF
    hfig = figure('Position', [1,1,w,h], 'visible', 'off');
%     hfig = figure('visible', 'off');
    if para.ForceSameFrame
        motionIdx = i;
    else
        if motionList(motionIdx) < i
            motionIdx = min([motionIdx+1, length(motionList)]);
        end
    end
    subplot('Position', [0,0,1,1]);
    if grayScaleVid
        im = imresize(videoOrig(:,:,i),[h,w]);
    else
        im = imresize(videoOrig(:,:,:,i),[h,w]);
    end
    if ~isempty(para.vidRescale)
        im = imresize(im, para.vidRescale);
        [xgrid,ygrid] = meshgrid(xlist*para.vidRescale,ylist*para.vidRescale);
        vidRescale = para.vidRescale;
    else
        [xgrid,ygrid] = meshgrid(xlist,ylist);
        vidRescale = 1;
    end
    imshow(im); hold on;
    if isempty(para.abs_quiver_scale)
        hQuiver = quiver(xgrid,ygrid,motionScaledVid(ylist,xlist,1,motionIdx), ...
            motionScaledVid(ylist,xlist,2,motionIdx), 1, 'LineWidth', para.quiver_width,...
            'Color', para.quiver_color); hold off;
    else
        tscale = para.abs_quiver_scale * para.quiver_step * vidRescale;
        hQuiver = quiver(xgrid,ygrid,tscale*motionScaledVid(ylist,xlist,1,motionIdx), ...
            tscale*motionScaledVid(ylist,xlist,2,motionIdx), 0, 'LineWidth', para.quiver_width,...
            'Color', para.quiver_color); hold off;
    end
    adjust_quiver_arrowhead_size(hQuiver, para.quiver_headsize);
%     truesize(hfig,[h,w]); 
    axis off;
%     quiverFrame = getframe(hfig);
    quiverFrame = export_fig(hfig, '-nocrop');
    writeVideo(writerQuiverMotion, quiverFrame);
    close(hfig);
    if mod(i,10) == 0
        fprintf(1, 'frame %d/%d is done\n', i, nF);
    end
end
if ~isempty(outQuiverFile)
    close(writerQuiverMotion);
end

end