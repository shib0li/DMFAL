function y=PhiU0(x,i,N,a,b)
% hat function and u0 product operation 
%
% Modifications:
% 26-May-2015, WeiX, first edition 
% 27-May-2015, WeiX, second edition, improve for vector operation
%%
    y= U0(x,a,b).*FEM_HatFunc(i,N,x);
    
    
end
