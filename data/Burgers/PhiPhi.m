function y=PhiPhi(x,i,j,N)
% hat function product operation 
%
% Modifications:
% 26-May-2015, WeiX, first edition 
% 27-May-2015, WeiX, second edition, improve for vector operation
%%
    y= FEM_HatFunc(i,N,x ).*FEM_HatFunc(j,N,x );
    
%     a=FEM_HatFunc3(i,N,x );
%     b=FEM_HatFunc3(j,N,x );
%     y=a*b;
    
    
    
end



