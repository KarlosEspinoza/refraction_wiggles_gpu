function [vu, vA] = optflow(vid, varargin)

    % [vu, vA] = optflow(vid, varargin)
    % 
    % Input:
    %   vid: h*w*nF - input gray scale video
    % 
    % Output:
    %   vu: h*w*2*(nF-1) - mean of motion vector
    %   vA: cell(1,nF-1), each (h*w*2)*(h*w*2) - precision matrix of motion vector

    para.alpha1 = 1;
    para.alpha2 = 1;
    para.spatialSigma = [];
    para.tempSigma = [];
    para.adpSpatialSigma = 2;
    para.localWinsize = 30;
    para.localPadsize = round(para.localWinsize/2);
    para.normIt = true;
    para.verbose = 0;
    para.sorMethod = 0; %-1: don't calculate vu, 0: no sor; 1: sor by dongarra; 2: sor by deqing
    para.sorW = 1.7;
    para.sorIter1 = 10;
    para.sorIter2 = 30;
    para.adpW = false;
    para = propval(varargin, para);

    % compute dI/dx
    f = [1,0,-1];
    flen = 2; flen2 = flen/2;
    alpha1Sqrt = sqrt(para.alpha1);
    alpha2Sqrt = sqrt(para.alpha2);
    % Spatial & temporal blur the input video sequence
    if ~isempty(para.spatialSigma)
        sw = ceil(para.spatialSigma*2);
        [tx,ty] = meshgrid(-sw:sw, -sw:sw);
        fxy = exp(-(tx.^2+ty.^2)/para.spatialSigma^2);
        fxy = fxy / sum(fxy(:));
        vid = convnFFT(padarray(vid,[sw,sw,0],'symmetric'),fxy,'valid');
    end
    if nargout == 1
        outVaIf = false;
    else
        outVaIf = true;
    end
    Ix = padarray(convn(vid, f, 'valid'),[0,flen2,0], 'replicate');
    Iy = padarray(convn(vid, f', 'valid'),[flen2,0,0], 'replicate');
    vu = 0;
    vA = 0;

    % use random values to replace small values
    tmask = abs(Ix)<1e-3;
    Ix(tmask) = 1e-3*(1+rand(nnz(tmask),1));
    tmask = abs(Iy)<1e-3;
    Iy(tmask) = 1e-3*(1+rand(nnz(tmask),1));
    clear tmask
    
    % compute dI/dt 
    It = vid(:,:,2:end) - vid(:,:,1:end-1);


    % normalize dI/dt
    if para.normIt
        sdIt = sqrt(mean(mean(It.^2,1),2));
        medianIt = median(sdIt);
        It = bsxfun(@times, It, medianIt./sdIt);
    end
    if para.adpW
        sw = para.adpSpatialSigma*2;
        fs2 = fspecial('gaussian', [2*sw+1,2*sw+1], para.adpSpatialSigma);
    else
        fs2 = []; sw = [];
    end

    [h,w,nF] = size(vid); nF = nF-1;
    vu = zeros(h,w,2,nF-1);
    if outVaIf
        vA = cell(1,nF-1);
    end
    parfor t=1:nF
        if para.adpW
            avgIx = double(sqrt(padarray(convn(abs(Ix(:,:,t)), fs2, 'valid'),[sw,sw],'replicate')));
            avgIy = double(sqrt(padarray(convn(abs(Iy(:,:,t)), fs2, 'valid'),[sw,sw],'replicate')));
        end

        % Solve the linear equation to get v
        A1i = reshape(repmat(1:h*w,2,1),1,h*w*2);
        A1j = [1:h*w; h*w+1:h*w*2];
        A1s = alpha1Sqrt*double(reshape([reshape(Ix(:,:,t),1,h*w);reshape(Iy(:,:,t),1,h*w)],1,h*w*2));
        A1 = sparse(A1i, A1j, A1s, h*w, h*w*2);

        A1i=[]; A1j=[]; A1s=[]; % clear variables
        if para.sorMethod >= 0 
            b = -alpha1Sqrt * double(reshape(It(:,:,t),h*w,1));
        end

        % Solve A2y
        [x,y] = meshgrid(1:w,2:h); idx = sub2ind([h,w],y(:),x(:))';
        A2yi = reshape(repmat(1:(h-1)*w,2,1),1,(h-1)*w*2);
        A2yj = reshape([idx;idx-1],1,(h-1)*w*2);
        nA2y = (h-1)*w;
        if para.adpW
            tmp = reshape(avgIx(1:h-1,:),1,nA2y); A2dvxdy = reshape(alpha2Sqrt*[tmp;-tmp],1,nA2y*2);
            tmp = reshape(avgIy(1:h-1,:),1,nA2y); A2dvydy = reshape(alpha2Sqrt*[tmp;-tmp],1,nA2y*2);
            A2y = [sparse(A2yi, A2yj, A2dvxdy, nA2y, h*w*2); sparse(A2yi, A2yj+h*w, A2dvydy, nA2y, h*w*2)];
            A2dvydy = []; A2dvydy = [];
        else
            A2ys = reshape(repmat([alpha2Sqrt;-alpha2Sqrt],1,nA2y),1,nA2y*2);
            A2y = [sparse(A2yi, A2yj, A2ys, nA2y, h*w*2); sparse(A2yi, A2yj+h*w, A2ys, nA2y, h*w*2)];    
            A2ys=[];
        end
        A2yi=[]; A2yj=[]; % clear variables A2y_ij

        % Solve A2x
        [x,y] = meshgrid(2:w,1:h); idx = sub2ind([h,w],y(:),x(:))';
        A2xi = reshape(repmat(1:h*(w-1),2,1),1,h*(w-1)*2);
        A2xj = reshape([idx;idx-h],1,h*(w-1)*2);
        nA2x = h*(w-1);
        if para.adpW
            tmp = reshape(avgIx(:,1:w-1,:),1,nA2x); A2dvxdx = reshape(alpha2Sqrt * [tmp;-tmp],1,nA2x*2);
            tmp = reshape(avgIy(:,1:w-1,:),1,nA2x); A2dvydx = reshape(alpha2Sqrt * [tmp;-tmp],1,nA2x*2);
            A2x = [sparse(A2xi, A2xj, A2dvxdx, nA2x, h*w*2);sparse(A2xi, A2xj+h*w, A2dvydx, nA2x, h*w*2)];
            A2dvxdx = []; A2dvydx = [];
        else
            A2xs = reshape(repmat([alpha2Sqrt;-alpha2Sqrt],1,h*(w-1)),1,h*(w-1)*2);
            A2x = [sparse(A2xi, A2xj, A2xs, nA2x, h*w*2); sparse(A2xi, A2xj+h*w, A2xs, nA2x, h*w*2)];
            A2xs=[];
        end
        A2xi=[]; A2xj=[]; % clear variables A2x_ij

        % Solve A
        A = [A1;A2x;A2y];
        A1=[]; A2y=[]; A2x=[]; % clear variables A1 A2y A2x
        %     vvec = lsqr(A,[b;zeros(nA2x*2+nA2y*2,1)]);
        if para.sorMethod > 0
            Ap = A'; A1 = Ap*A; 
            b1 =  Ap*[b;zeros(nA2x*2+nA2y*2,1)];
            if para.sorMethod == 1
                vvec = sor_dongarra(A1, ones(h*w*2,1), b1, para.sorW, para.sorIter1);
            elseif para.sorMethod == 2
                vvec = sor_deqing(A1, b1, para.sorW, para.sorIter1, 1e-2, ones(h*w*2,1));
            else
                error('Incorrect sorMethod index');
            end
            Ap = [];
            vvec(abs(vvec)>0.1) = 0;
            if para.sorMethod == 1
                vvec = sor_dongarra(A1, vvec, b1, para.sorW, para.sorIter2);
            elseif para.sorMethod == 2
                vvec = sor_deqing(A1, b1, para.sorW, para.sorIter2, 1e-2, vvec);
            else
                error('Incorrect sorMethod index');
            end
        elseif para.sorMethod == 0
            vvec = A \ [b;zeros(nA2x*2+nA2y*2,1)];
        end
        if para.sorMethod >= 0 
            b=[];
            vu(:,:,:,t) = reshape(vvec, h, w, 2);
            vvec=[];
        end

        % Solve for variance
        if outVaIf
            vA{t} = A'*A * 1e6;
        end
        A = [];
        if para.verbose == 1
            fprintf('frame %d is done\n',t);
        end
    end
end
