function wheelIm = computeColorWheel(imhalfsize)

if ~exist('imhalfsize', 'var')
    imhalfsize = 100;
end

[x,y] = meshgrid(-imhalfsize:imhalfsize, -imhalfsize:imhalfsize);
x = x/(imhalfsize*sqrt(2)); y = y/(imhalfsize*sqrt(2));
wheelIm = computeColor(x,y);

end