clc;clear all;close all;

% m = -1;
% buffer = '../__buffer__/test.mat';
% 
% ground_fidelity = 100;
% fidelity_list = [25,50,75];
% 
% if m == -1
%     fid = ground_fidelity;
% else
%     fid = fidelity_list(m+1); 
% end

% Re_list = X*500+10;
% nSample = size(X, 1);
% %re_List = rand(nSample,1)*500+10;
% g_fid = 128; %fidelity used to calc groud-truth
% interp_dim = 128;

% xg = linspace(0,1,fid+1);
% yg = linspace(0,Paras.t_end,fid+1);
% xg_interp = linspace(0,1,interp_dim);
% yg_interp = linspace(0, Paras.t_end,interp_dim);
% %% Main
% 
% Rec_Y = [];
% for j=1:nSample
%     Paras.Re = Re_list(j);
%     [Y,time,Time_Ode_solver]=Burger1D_FEM_DBC_SolverF(Paras);
%     Rec_Y(:,:,j) = Y;
%     Y_interp = interp2(xg,yg',Y,xg_interp,yg_interp');
%     Rec_Y_interp(:, :, j) = Y_interp;
% end

%
fid = 50;
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