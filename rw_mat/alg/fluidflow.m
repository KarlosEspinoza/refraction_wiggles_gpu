function [umean,uvar] = fluidflow(vu, vA, varargin)

% function u = fluidflow(vu, vA, varargin)
% 
% Input:
%   v: h*w*2*nF - mean of motion vector
%   vA: h*w*2*2*nF - variance of motion vector
%
% Output:
%	umean: h*w*2*(nF-1), the mean of fluid flow
%   uvar: h*w*3*(nF-1), the variance of fluid flow
% 
% Parameters:
%   beta2, beta3, nOuterIter, nInnerIter, startSigma, endSigma, maxV,
%   verbose, timeWin, varSigma, sorMethod
% 
% Objective function:
%   \min_u (pvpx*ux + pvpy*uy + pvpy)*Sigma^-1*(pvpx*ux + pvpy*uy + pvpy)
%           + beta2 * (|ux|^2 + |uy|^2) + beta3 * |u|^2

para.beta2 = 1;
para.beta3 = 1e-7;
para.nOuterIter = 4;
para.nInnerIter = 1;
para.startSigma = 8;
para.endSigma = 1;
para.maxV = 50;
para.verbose = 1;
para.timeWin = 2;
para.varSigma = 3;
para.sorMethod = 0; %0: no sor; 1: sor by dongarra
para.outFrameList = [];
para.outFrameValidOnly = true;
para.clearTempMem = true;
para = propval(varargin, para);

fdx = [1,0,-1]; fsqueeze = [fdx(1),fdx(3)];
flen = 2; flen2 = flen/2; fidx = [-1;1];
sigmaList = exp(linspace(log(para.startSigma),log(para.endSigma),para.nOuterIter));
% Solve for u
[h,w,~,nF] = size(vu);
if isempty(para.outFrameList)
    if para.outFrameValidOnly
        para.outFrameList = para.timeWin+1:nF-1-para.timeWin;
    else
        para.outFrameList = 1:nF-1;
    end
end
nFrameOut = length(para.outFrameList);
umean = zeros(h,w,2,nFrameOut,'single');
uvar = zeros(h,w,3,nFrameOut,'single');
vcell = cell(1,nFrameOut);
for tidx=1:nFrameOut
    t = para.outFrameList(tidx);
    frameList = max(1,t-para.timeWin):min(nF, t+1+para.timeWin);
    vcell{tidx} = single(vu(:,:,:,frameList));
end
clear vu
sw = para.varSigma*3;
fs = fspecial('gaussian', [sw*2+1,sw*2+1], para.varSigma);
fsWeight = ones([h,w]);
fsWeight = convnFFT(fsWeight, fs, 'same');
fs = fs/(fs(sw+1,sw+1));
if para.sorMethod > 0
    iterTime = zeros(1,para.nInnerIter);
    iterTime(end) = 100;
    for iter = para.nInnerIter-1:-1:1
        iterTime(iter) = round(iterTime(iter+1) * 0.75);
    end
else
    iterTime=[];
end
if isempty(vA)
    noOptVar = true;
    vA = cell(1,nFrameOut);
else
    noOptVar = false;
    vA = vA(para.outFrameList);
end

parfor tidx=1:nFrameOut
% for tidx=1:nFrameOut
    t = para.outFrameList(tidx);
    tumean = zeros(h,w,2,'single');    
    
    % construct A matrix
    [xmask,ymask] = meshgrid(1:w,2:h); idx = sub2ind([h,w],ymask(:),xmask(:))';
    Dyi = reshape(repmat(1:(h-1)*w,2,1),1,(h-1)*w*2);
    Dyj = reshape([idx;idx-1],1,(h-1)*w*2);
    Dys = reshape(repmat([1;-1],1,(h-1)*w),1,(h-1)*w*2);
    Dy = sparse(Dyi,Dyj,Dys,(h-1)*w,h*w);
    if para.clearTempMem
        Dyi = []; Dyj = []; Dys = []; 
    end
    [xmask,ymask] = meshgrid(2:w,1:h); idx = sub2ind([h,w],ymask(:),xmask(:))';
    Dxi = reshape(repmat(1:h*(w-1),2,1),1,h*(w-1)*2);
    Dxj = reshape([idx;idx-h],1,h*(w-1)*2);
    Dxs = reshape(repmat([1;-1],1,h*(w-1)),1,h*(w-1)*2);
    Dx = sparse(Dxi,Dxj,Dxs,h*(w-1),h*w);
    Dxi = []; Dxj = []; Dxs = []; 
    A2 = para.beta2*(Dy'*Dy + Dx'*Dx);
    A2 = blkdiag(A2, A2);
    if para.clearTempMem
        Dx = []; Dy = [];
    end
    A3 = sparse(1:h*w*2,1:h*w*2,repmat(para.beta3,1,h*w*2),h*w*2,h*w*2);
    [xdx,ydx] = meshgrid(2:w-1, 1:h); xdx = single(xdx); ydx = single(ydx);
    [xdy,ydy] = meshgrid(1:w, 2:h-1); xdy = single(xdy); ydy = single(ydy);
    [xGrid,yGrid] = meshgrid(1:w,1:h); xGrid = single(xGrid); yGrid = single(yGrid);
    frameList = max(1,t-para.timeWin):min(nF, t+1+para.timeWin);
    
    for outIter = 1:para.nOuterIter
        % Blur the input video
        sw = sigmaList(outIter);
        bandWidth = ceil(sw*3);
        f = fspecial('gaussian', ceil([bandWidth*2+1,bandWidth*2+1]), sw);
        psize = [bandWidth,bandWidth];
        vt = single(convnFFT(padarray(vcell{tidx}, psize,'replicate'),f,'valid'));
%         dvxGrid = convn(vt1, fdx, 'valid'); % Push down?
%         dvyGrid = convn(vt1, fdx', 'valid');
        
        for inIter = 1:para.nInnerIter
            % Calculate the derivative
            [x,y] = meshgrid(1:w,1:h);
            tmask = (x >= 3) & (x <= w-2) & (y >= 3) & (y <= h-2);
            x = single(x + tumean(:,:,1)); y = single(y + tumean(:,:,2));
            mask = (x >= 3) & (x <= w-2) & (y >= 3) & (y <= h-2) & tmask;
            xmask = x(mask); ymask = y(mask);
            
            BSB = sparse(h*w*2,h*w*2); BSc = sparse(h*w*2,1);
            for tf = 1:length(frameList)-1
                dvxGrid = single(convn(vt(:,:,:,tf), fdx, 'valid'));
                dvyGrid = single(convn(vt(:,:,:,tf), fdx', 'valid'));
                B11s = zeros(h,w); B12s = zeros(h,w); B21s = zeros(h,w); B22s = zeros(h,w);
                B11s(mask) = interp2(xdx, ydx, dvxGrid(:,:,1), xmask, ymask); 
                B12s(mask) = interp2(xdy, ydy, dvyGrid(:,:,1), xmask, ymask);
                B21s(mask) = interp2(xdx, ydx, dvxGrid(:,:,2), xmask, ymask); 
                B22s(mask) = interp2(xdy, ydy, dvyGrid(:,:,2), xmask, ymask);
                B = [sparse(1:h*w,1:h*w,B11s,h*w,h*w), sparse(1:h*w,1:h*w,B12s,h*w,h*w);
                    sparse(1:h*w,1:h*w,B21s,h*w,h*w), sparse(1:h*w,1:h*w,B22s,h*w,h*w)];
                if para.clearTempMem
                    B11s = []; B12s = []; B21s = []; B22s = []; 
                end
                
                vt1x = vt(:,:,1,tf); vt1y = vt(:,:,2,tf);
                c1 = zeros(h,w); c2 = zeros(h,w);
                c1(mask) = interp2(xGrid, yGrid, vt(:,:,1,tf+1), xmask, ymask) - vt1x(mask);
                c2(mask) = interp2(xGrid, yGrid, vt(:,:,2,tf+1), xmask, ymask) - vt1y(mask);
                if para.clearTempMem
                    vt1x=[]; vt1y=[];
                end
                
                if noOptVar
                    BA = B';
                else
                    BA = B' * vA{tidx};
                end
                BSB = BSB + BA*B; BSc = BSc + BA*[c1(:);c2(:)];
                if para.clearTempMem
                    BA = []; B = []; c1 = []; c2 = [];
                end
            end
            tuvec = double(reshape(tumean,h*w*2,1));
            if para.sorMethod == 1
                uvec = - sor_dongarra(BSB + A2 + A3, ones(h*w*2,1), BSc + A2*tuvec + para.beta3*tuvec, ...
                    1.97, iterTime(inIter));
            elseif para.sorMethod == 2
                uvec = - sor_deqing(BSB + A2 + A3, BSc + A2*tuvec + para.beta3*tuvec, ...
                    1.97, iterTime(inIter), 1e-2, ones(h*w*2,1));
            elseif para.sorMethod == 0
                uvec = - ((BSB + A2 + A3) \ (BSc + A2*tuvec + para.beta3*tuvec));
            else
                error('Incorrect sorMethod index');
            end
            if para.clearTempMem
                tuvec = []; xmask = []; ymask = []; tmask = []; mask = [];
                BSB = []; BSc = [];
            end
            tumean = tumean + reshape(uvec, h, w, 2);
            tumean(tumean > para.maxV) = para.maxV;
            tumean(tumean < -para.maxV) = -para.maxV;
            fprintf('frame %d, outIter = %d, inIter = %d\n',t, outIter, inIter);
        end
    end
    
    % Calculate the variance
    B = sparse(h*w*2,h*w*2);
    for tf = 1:length(frameList)-1
        dvxGrid = padarray(convn(vt(:,:,:,tf), fdx, 'valid'),[0,1,0],'replicate');
        dvyGrid = padarray(convn(vt(:,:,:,tf), fdx', 'valid'),[1,0,0],'replicate');
        B = B + [sparse(1:h*w,1:h*w,double(dvxGrid(:,:,1)),h*w,h*w), sparse(1:h*w,1:h*w,double(dvyGrid(:,:,1)),h*w,h*w);
            sparse(1:h*w,1:h*w,double(dvxGrid(:,:,2)),h*w,h*w), sparse(1:h*w,1:h*w,double(dvyGrid(:,:,2)),h*w,h*w)];
    end
    if noOptVar
        B = B' * B; len = size(B,1);
    else
        B = B' * vA{tidx} * B; len = size(B,1);
    end
    B11 = single(reshape(full(B(1:(h*w*2+1):sub2ind([h*w*2,h*w*2],h*w,h*w))),[h,w]));
    B12 = single(reshape(full(B(sub2ind([h*w*2,h*w*2],1,h*w+1):(h*w*2+1):sub2ind([h*w*2,h*w*2],h*w,h*w*2))),[h,w]));
    B21 = single(reshape(full(B(sub2ind([h*w*2,h*w*2],h*w+1,1):(h*w*2+1):sub2ind([h*w*2,h*w*2],h*w*2,h*w))),[h,w]));
    B22 = single(reshape(full(B(sub2ind([h*w*2,h*w*2],h*w+1,h*w+1):(h*w*2+1):sub2ind([h*w*2,h*w*2],h*w*2,h*w*2))),[h,w]));
    B11 = bsxfun(@rdivide, convnFFT(B11, fs, 'same'), fsWeight);
    B12 = bsxfun(@rdivide, convnFFT(B12, fs, 'same'), fsWeight);
    B21 = bsxfun(@rdivide, convnFFT(B21, fs, 'same'), fsWeight);
    B22 = bsxfun(@rdivide, convnFFT(B22, fs, 'same'), fsWeight);
    invdetB = (B11.*B22 - B12.*B21).^(-1);
    uvar(:,:,:,tidx) = cat(3, invdetB.*B22, -invdetB.*B21, invdetB.*B11);
    
    if para.clearTempMem
        B11 = []; B12 = []; B21 = []; B22 = []; invdetB = []; B = [];
        A2 = []; A3 = []; 
        dvdtIter = []; dvdxIter = []; dvdyIter = []; dvxGrid = []; dvyGrid = []; 
        xdx = []; ydx = []; xdy = []; ydy = []; xGrid = []; yGrid = [];
    end
    if para.verbose == 1
        fprintf('frame %d\n',t);
    end
    umean(:,:,:,tidx) = tumean;
    if para.clearTempMem
        tumean = [];
    end
end



end




































