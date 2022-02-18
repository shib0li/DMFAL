% function Burger1D_FEM_DBC_MOR_SolverF_Demo(Paras,U)
%
% Burgers equation 1D case, finite element method, Dirichlet boundary
% conditions & homeogeneous B.C, solver.

% Problem model:
% [u(t,x)]_t+1/2[u^2(t,x)]_x-q [u(t,x)]_xx=f(t,x), x \in (0,1),t>0;

% Boundary Condition (Dirichlet):
% u(t,0)=0; u(t,1)=0;

% Initial Conditions:
% u(0,x)=u_0(x);

% close all
 clear;

%% Setup
% ------------------Problem Parameters------------------------------------- 
Paras.Re=50;    % Reynolds Number
% v=1/Re;     % viscosity

Paras.u0a=1; 
Paras.u0b=1; 
% ------------------Solver Parameters--------------------------------------
Paras.n=128;           % Total Spatial elements
Paras.t_end=2;        % End time
Paras.t_n=100;        % Number of time step 
% Paras.t=0:(t_end/t_n):t_end; % time sequence (t=0 does not account 1 step)

% solver = 'ode45';
% options = odeset('RelTol',1e-6,'AbsTol',1e-10);

approximate_degree=1;

% ------------------Calculating temp variable----------------------------- 
% h=1/n;      % space step size
% x = 0:h:1;  % coordinate sequence

%% Main 

% Normal solver
[Y1,T1,Time_Ode_solver]=Burger1D_FEM_DBC_SolverF(Paras);

% POD
Y1_inter=Y1(2:end-1,:);
[U,S,~]=svd(Y1_inter);  % U*S*V'=Rec_X
U=U(:,1:approximate_degree);
eigenvalues=diag(S);

% MOR
[Y2,T2,Time_Ode_MORsolver]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U);







% Ploting
figure(1)
meshc(Y1)
title(sprintf('FEM Solution. Time cost= %0.3f',Time_Ode_solver))

figure(2)
meshc(Y2)
title(sprintf('MOR FEM Solution. Time cost= %0.3f',Time_Ode_MORsolver))


h=1/Paras.n;      % space step size
x = 0:h:1;  % coordinate sequence
for i = 1:Paras.t_n+1
    figure(3)
    
    subplot(2,1,1)
    plot(x,Y1(:,i))
    axis([0,1,0,1]);
    title(sprintf('FEM Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))
    
    subplot(2,1,2)
    plot(x,Y2(:,i))
    axis([0,1,0,1]);
    title(sprintf('MOR FEM Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))   
    
    F(i) = getframe;
    
end

% movie(F,2)
