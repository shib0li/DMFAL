clc;clear all;close all;

%
fid = 80;
X = rand(3,2); %input in [0, 1]
infl = X(:,1)*25+0.5;
thetaa = X(:,2)*(pi/2);
par = [floor(infl),thetaa]';
nelx=fid;nely=fid;volfrac=0.4;penal=3;rmin=1.5;
%nelx=fid;nely=fid;volfrac=0.4;penal=3;rmin=1.5;
%infl=(rand(1,nsample))*25+0.5;
%thetaa=rand(1,nsample)*(pi/2);
%par=[floor(infl);thetaa];
%load par.mat;
nsample = size(X, 1);
Y = zeros(fid, fid, nsample);
for i=1:nsample
    mag=1;theta=par(2,i);Ey=1;volfrac=0.4;rmin=1.5;eno=par(1,i)*2;
    tic;
    [sol,obj,itr]=LBracket_generator(nelx,nely,volfrac,penal,1.5,theta,eno);
    time_run(i)=toc;
    Y(:, :, i) = sol;
end
data = [];
data.X = X;
data.Y = Y;
save('new_samples.mat', 'data');
time_run

