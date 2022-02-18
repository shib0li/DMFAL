function [dy]=Burgers1D_DBC_FEM_ODE_func(t,y,M,B,C,F,v)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Burgers' equation 1D, finite element method, Dirichlet boundary 
% conditions, solver.
% 
% Function inputs:
%
% t : current time
% y : current y
% v : viscosity
%
%
% Function outputs:
% out y : computed derivative
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %F = RHS F(f, in t);
    % For the moment assume F=0;

    dy = M\ (F-(1/2)*B*y.^2-v*C*y);

end