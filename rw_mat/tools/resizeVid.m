function vidnew = resizeVid(vid, input2, varargin)

% function vidnew = resizeVid(vid, input2)
% 
% Input
%   vid: h*w*nF
%   input2: 
%       - length(input2)=1: input2 = scale
%       - length(input2)=2: input2 = vidsize
% 
% Parameter:
%   method

para.method = 'bicubic';
para = propval(varargin, para);

if length(input2) == 1
    scale = input2;
    vidsize = size(imresize(vid(:,:,1), scale));
else
    vidsize = input2;
end

[~,~,nF] = size(vid);
sizeVidnew = size(vid);
sizeVidnew(1:2) = vidsize;
if isa(vid, 'logical')
    vidnew = true(sizeVidnew);
else
    vidnew = zeros(sizeVidnew, class(vid));
end
for i=1:nF
    vidnew(:,:,i) = imresize(vid(:,:,i), vidsize, para.method);
end

% if length(size(vid)) == 4
% 	[~,~,nC,nF] = size(vid);
% 	vidnew = zeros([sh,sw,nC,nF], class(vid));
% 	for i=1:nF
% 		vidnew(:,:,:,i) = imresize(vid(:,:,:,i), scale);
% 	end
% else
% 	[~,~,nF] = size(vid);
% 	vidnew = zeros([sh,sw,nF], class(vid));
% 	for i=1:nF
% 		vidnew(:,:,i) = imresize(vid(:,:,i), scale);
% 	end
% end

end

