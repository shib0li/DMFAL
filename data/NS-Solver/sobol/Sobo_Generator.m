function [X] = Sobo_Generator(Num,rang)
% Description:
% Sobo sequence experiment data set generation. 
% This use sobol package for experiment design purpose.All credic goes to the original author.
%
% Synopsis:
% [X] = Sobo_Generator(rang)
%
% Input: 
%       Num               % Number of total data point
%       rang              % Original dataset [2 X dimension ] matrix.
%                         [1,x] [1,x] are up and down limit of x dimension respectively.
% Output: 
%       X                 % Result dataset [Num x dimension] 
%
% Pcakage Require: sobol
%
% See also 
%
%  Modification:
%  WeiX, Nov 26th 2014, First Edition

[~, dimension ]=size(rang);

X=i4_sobol_generate ( dimension, Num, 0 );
X=X';

for i=1:dimension
    uplimit=rang(1,i);
    downlimit=rang(2,i);
    X(:,i)=X(:,i)*(uplimit-downlimit)+ones(Num,1)*downlimit;
end