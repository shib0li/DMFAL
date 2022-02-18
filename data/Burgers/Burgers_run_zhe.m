%multiFedility_data_Gen_v02

% close all
clear all; close all;
rng(1234);
%input
X = rand(5, 1);
fid = 32;
Re_list = X*500+10;
nSample = size(X, 1);
%re_List = rand(nSample,1)*500+10;
g_fid = 128; %fidelity used to calc groud-truth
interp_dim = 100;

%parameter setting
Paras = {};
Paras.Re = 1;
Paras.u0a = 1;
Paras.u0b = 0.5; %funny, setting it to 0 always return 0s
Paras.n = fid;
Paras.t_n = fid;
Paras.t_end = 3; %t in [0, 3]

xg = linspace(0,1,fid+1);
yg = linspace(0,Paras.t_end,fid+1);
xg_interp = linspace(0,1,100);
yg_interp = linspace(0, Paras.t_end,100);
%% Main
Rec_Y = [];
for j=1:nSample
    Paras.Re = Re_list(j);
    [Y,time,Time_Ode_solver]=Burger1D_FEM_DBC_SolverF(Paras);
    Rec_Y(:,:,j) = Y;
    Y_interp = interp2(xg,yg',Y,xg_interp,yg_interp');
    Rec_Y_interp(:, :, j) = Y_interp;
end
data = {};
data.X = X;
data.Y = Rec_Y;
data.Y_interp = Rec_Y_interp;
save('new_samples.mat', 'data');

    


