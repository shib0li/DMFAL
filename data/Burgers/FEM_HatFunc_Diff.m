function  Phi  = FEM_HatFunc_Diff(i,N,x )
% Differential euqation of Hat_Function3
%
% Modifications:
% 27-May-2015, WeiX, first edition 

h=1/N;

% Phi=heaviside(x-i*h)/h;

Phi=1/h*(x<=i*h)-1/h*(x>i*h);
% for matrix computation
Phi=Phi.*(x<h*(i+1));
Phi=Phi.*(x>h*(i-1));

end

