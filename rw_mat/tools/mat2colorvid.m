function vid = mat2colorvid(mat, varargin)

% function vid = mat2colorvid(mat, varargin)
% 
% Parameters:
%   limits, clevel, map

para.limits = [];
para.clevel = 256;
para.map = [];
para = propval(varargin, para);

if isempty(para.limits)
    para.limits = [max(mat(:)), min(mat(:))];
end
if isempty(para.map)
    clevel = para.clevel;
%     cmap = hsv(round(para.clevel * 1.33));
%     cmap = cmap(end-clevel+1:end,:);
    cmap = jet(para.clevel);
else
    clevel = size(para.map,1);
    cmap = para.map;
end

[imh, imw, nf] = size(mat);
vid = zeros(imh, imw, 3, nf);
for k=1:nf
    vid(:,:,:,k) = ind2rgb(gray2ind(mat2gray(mat(:,:,k), double(para.limits)), clevel), cmap);
end

end

