function writeVid(mat, filepath, framerate, profile, quality)

% function writeVid(mat, filepath, framerate, profile)
% 
% Input:
%   mat: matrix
%   filepath: output filepath

if ~exist('framerate', 'var')
    framerate = 10;
end
if ~exist('profile', 'var')
	profile = 'Motion JPEG AVI';
end
if ~exist('quality', 'var')
	quality = 85;
end

if isa(mat, 'uint16')
    mat = double(mat)/2^16;
elseif isa(mat, 'uint8')
    mat = double(mat)/2^8;
end

writerObj = VideoWriter(filepath, profile);
if ~strcmp(profile, 'Uncompressed AVI')
    writerObj.Quality = quality;
end
writerObj.FrameRate = framerate;
open(writerObj);
if length(size(mat))==3
    nF = size(mat,3);
    grayIf = true;
else
    nF = size(mat,4);
    grayIf = false;
end
for i=1:nF
    if grayIf
        im = mat(:,:,i);
    else
        im = mat(:,:,:,i);
    end
    im(im > 1) = 1;
    im(im < 0) = 0;
	writeVideo(writerObj, im);
end
close(writerObj);

end


