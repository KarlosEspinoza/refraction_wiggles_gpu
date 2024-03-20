function mat = vid2mat(vidfile, varargin)

% function mat = vid2mat(vidfile)
% 
% Parameters:
%   frameRange
%   grayDoubleConvertIf

para.frameRange = [];
para.grayDoubleConvertIf = false;
para = propval(varargin, para);
vidReader = VideoReader(vidfile);
if ~para.grayDoubleConvertIf
    if isempty(para.frameRange)
        mat = read(vidReader,[1,inf]);
    else
        mat = read(vidReader,para.frameRange);
    end
else
    if isempty(para.frameRange)
        read(vidReader, inf);
        para.frameRange = [1,vidReader.NumberOfFrames];
    end
    for i=para.frameRange(1):para.frameRange(2)
        im = rgb2gray(im2double(read(vidReader,[i,i])));
        if i==para.frameRange(1)
            mat = zeros([size(im),para.frameRange(2)-para.frameRange(1)+1]);
        end
        mat(:,:,i-para.frameRange(1)+1) = im;
    end
end

end
