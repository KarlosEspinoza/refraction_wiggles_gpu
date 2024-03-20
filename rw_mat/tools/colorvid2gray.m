function grayvid = colorvid2gray(colorvid)

% function grayvid = colorvid2gray(colorvid)

[h,w,~,nF] = size(colorvid);
grayvid = zeros([h,w,nF], class(colorvid));
for i=1:nF
    grayvid(:,:,i) = rgb2gray(colorvid(:,:,:,i));
end

end