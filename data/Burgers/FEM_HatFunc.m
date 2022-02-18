function  Phi  = FEM_HatFunc(i,N,x )
% Description:
% Finite element widely used hat function as the basic function3
% a better way to express FEM hat function.(simpler, without 'if' operation
% and other logical operation, Thus can be used with ingegral function)
%
% WARMING: inputs should always satisfy: x>=0 && x<=1; 
%
% Synopsis:
% Phi  = FEM_HatFunc3(i,N,x )
%
% Input: 
%       i                 % index of the ith basic subfunction  i \in [0,N];
%       N                 % number of element. Total number of basic 
%                         % function =N+1.(including i=0) 
%                         %(normally N=number of element+1 in 1D case) 
%       x                 % variable x
% Output: 
%       Phi               % output 
% Pcakage Require:
%
% Modifications:
% 27-May-2015, WeiX, first edition 

h=1/N;

Phi=1-abs(x-i*h)/h;

% for matrix computation
Phi=Phi.*(x<h*(i+1));
Phi=Phi.*(x>h*(i-1));

%Phi=Phi.*(heaviside(h*(i-1))-heaviside(h*(i+1))) % A alternative solution.
%But wrong. Need improve.

% Equivalent to the following but capible for matrix computation.
% h=1/N;
% X=h:h:1;
% 
% if i ==0
%   
%     if ((x>=0) && (x<=X(1)))
%         Phi=(X(1)-x)/h;
%     else 
%         Phi=0;
%     end
%     
% elseif i==1    %Compromise for X(0)(=0) not access in MATLAB
%     if ((x>=0) && (x<=X(i)))   
%         Phi=(x-0)/h;    
%     elseif x>=X(i) && x<=X(i+1)
%         Phi=(X(i+1)-x)/h;
%     else
%         Phi=0;        
%     end        
%        
% elseif i>=1 && i<=N-1
%     
%     if x>X(i-1) && x<=X(i)   
%         Phi=(x-X(i-1))/h;     
%         
%     elseif x>=X(i) && x<=X(i+1)
%         Phi=(X(i+1)-x)/h;
%     else
%         Phi=0;        
%     end    
%     
% elseif i ==N
%     
%     if x>=X(N-1) && x<=X(N)
%         Phi=(x-X(N-1))/h;
%     else 
%         Phi=0;
%     end
% 
% end

end

