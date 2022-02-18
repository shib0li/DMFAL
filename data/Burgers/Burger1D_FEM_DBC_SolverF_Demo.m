% function Burger1D_FEM_DBC_SolverF_Demo
% Burgers equation 1D case, finite element method, Dirichlet boundary
% conditions & homeogeneous B.C, solver.Function.

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
Paras.Re=200;    % Reynolds Number
% v=1/Re;     % viscosity

Paras.u0a=1; 
Paras.u0b=1.5; 
% ------------------Solver Parameters--------------------------------------
Paras.n=64;           % Total Spatial elements
Paras.t_end=1;        % End time
Paras.t_n=64;        % Number of time step 
% Paras.t=0:(t_end/t_n):t_end; % time sequence (t=0 does not account 1 step)

% solver = 'ode45';
% options = odeset('RelTol',1e-6,'AbsTol',1e-10);

% ------------------Calculating temp variable----------------------------- 
% h=1/n;      % space step size
% x = 0:h:1;  % coordinate sequence

%% Main 
[Y,T,Time_Ode_solver]=Burger1D_FEM_DBC_SolverF(Paras);
% Paras.x = 2;
% [Y,T,Time_Ode_solver]=Burger1D_FEM_DBC_SolverF_v2(Paras);

% Ploting
figure(1)
meshc(Y)
title(sprintf('Full FEM Solution'));


h=1/Paras.n;      % space step size
x = 0:h:1;  % coordinate sequence
x = 0:1/Paras.n:1;  % coordinate sequence
Y_Maxi=max(Y(:));
Y_Mini=min(Y(:));

for i = 1:Paras.t_n+1
    figure(3)
    plot(x,Y(:,i))
    axis([0,1,Y_Mini,Y_Maxi]);
    title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))
    pause(0.1);  
end


