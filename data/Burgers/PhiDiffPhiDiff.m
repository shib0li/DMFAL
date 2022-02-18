function y=PhiDiffPhiDiff(x,i,j,N)
% diffential hat function product operation 
%
% Modifications:
% 27-May-2015, WeiX, first edition 
%%
    y= FEM_HatFunc_Diff(i,N,x ).*FEM_HatFunc_Diff(j,N,x );
    
%     a=FEM_HatFunc3(i,N,x );
%     b=FEM_HatFunc3(j,N,x );
%     y=a*b;
    
    
    
end