A = imread('9765.png');
A = rgb2gray(A);
A = A > 90;
imshow(A);