function y=U0(x,a,b)
% Initial condition function (time independent)
%
% Modifications:
% 27-May-2015, WeiX, first edition 
%%
%     y=x;
%     y=sin(pi*x);
%    y=sin(2*pi*x).*exp(x);
%      y=a*sin(b*pi*x);
%    y=a*exp(b*x);

%      y=sin(a*pi*x+b);
%     y=a*exp(-b*x).*sin(x/(2*pi));
    y=a*exp(-x * b).*sin(pi * x * 2 * b);
%     y=sin(a*pi*x+b*pi);
    
end

    