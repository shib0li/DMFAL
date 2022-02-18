%% ProjetMOR_Burger1D_FEM_Part1_TestHDM
%
% Modifications:
% 29-May-2015, WeiX, first edition 


% close all
 clear;

%% Setup
% ------------------Problem Parameters------------------------------------- 
% Paras.Re=100;    % Reynolds Number
% v=1/Re;     % viscosity

% rang=[500,10,10;1,0,0]; % setting parameter rang

% Paras.u0a=1; 
% Paras.u0b=2; 
% ------------------Solver Parameters--------------------------------------
Paras.n=128;           % Total Spatial elements
Paras.t_end=4;        % End time
Paras.t_n=200;        % Number of time step 

Paras.n_snap=20;      % ture number =Paras.n_snap+1. %WARMING:Paras.t_n/Paras.n_snap must =integer.
% Paras.t=0:(t_end/t_n):t_end; % time sequence (t=0 does not account 1 step)

% solver = 'ode45';
% options = odeset('RelTol',1e-6,'AbsTol',1e-10);

% approximate_degree=5;

% ------------------Calculating temp variable----------------------------- 
% h=1/n;      % space step size
% x = 0:h:1;  % coordinate sequence

% --------------------Sobol Sequence the parameters---------------------
Num_X=500;
% rang=[20,20,10;-20,-20,0]; 
rang=[100,0.5,0.5;10,0,0]; 

[X] = Sobo_Generator(Num_X,rang);
Index_snapshot=1:(Paras.t_n/Paras.n_snap):Paras.t_n+1;     %Shoube be adjusted accroding to Paras.dt & Paras.t_end;

h = waitbar(0,'TestHDM is working very hard for you');
for i = 1:Num_X
    Paras.Re=X(i,1);
    Paras.u0a=X(i,2);
    Paras.u0b=X(i,3);
    
    [y,T1,Time_Ode_solver]=Burger1D_FEM_DBC_SolverF(Paras);
    
    Y_Rec(:,:,i)=y; 
        
    y=y(:,Index_snapshot);
    Y(i,:)=y(:)';
    Time(i,1)=Time_Ode_solver;
    
    waitbar(i/Num_X);
%     waitbar(i/Num_X,'TestHDM is working very hard for you');
    
%     h=waitbar(i/Num_X,'TestHDM is working very hard for you');
%     h=waitbar(i/Num_X,sprintf('TestHDM is working very hard for you. %d%% done...',i/Num_X));
    
end
close(h);

%     save('ExpData11.mat','X','Y','Paras','rang');
    save('ExpDataV2_2.mat','X','Y_Rec','Time','Paras','rang');





