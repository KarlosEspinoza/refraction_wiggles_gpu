function writeVideo( vid, outFile, varargin)

% function writeVideo( vid, outFile, varargin)

para.rate = 25;
para.compressif = true;
para.Quality = 100;
para = propval(varargin, para);

% vw = VideoWriter(outFile, 'Archival');

if para.compressif
    vw = VideoWriter(outFile);
    vw.Quality = 100;
else   
    vw = VideoWriter(outFile, 'Uncompressed AVI');
end
vid(vid>1)=1;
vid(vid<0)=0;
vw.FrameRate = para.rate;
vw.open;    
vw.writeVideo(vid);
vw.close;
end

