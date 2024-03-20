function C = convnFFT(A, B, shape)

% function C = convnFFT(A, B, shape)
% A fast version of convn

if length(size(B)) > 2
    error('B must a matrix');
end

[h,w,nC] = size(A);
[hB,wB] = size(B);
hpad = h + hB; hpad2 = round(hpad/2);
wpad = w + wB; wpad2 = round(wpad/2);
Bf = fft2(padarray(B,[h,w],0,'post'));

sizeC = size(A);
if strcmp(shape, 'same')
    h2 = round(h/2); w2 = round(w/2);
    cropY = hpad2-h2+(0:h-1);
    cropX = wpad2-w2+(0:w-1);
    C = zeros(sizeC,class(A));
elseif strcmp(shape, 'valid')
    hv = h-hB+1; wv = w-wB+1;
    hv2 = round(hv/2); wv2 = round(wv/2); 
    cropY = hpad2-hv2+(0:hv-1);
    cropX = wpad2-wv2+(0:wv-1);
    sizeC(1:2) = [hv,wv];
    C = zeros(sizeC,class(A));
elseif strcmp(shape, 'full')
    cropY = 1:hpad-1;
    cropX = 1:wpad-1;
    sizeC(1:2) = [hpad-1,wpad-1];
    C = zeros(sizeC,class(A));
end

for t=1:nC
    Af = fft2(padarray(A(:,:,t),[hB,wB],0,'post'));
    Cf = Af.*Bf;
    Cpad = real(ifft2(Cf));
    C(:,:,t) = Cpad(cropY, cropX);
end

end

