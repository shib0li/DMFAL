%Demo_FEM_HatFunc
%
% Modifications:
% 27-May-2015, WeiX, first edition 


clear 

n=10;   %number of basic functions/Elements
n_x=100;
X= linspace(0,1,n_x);

for i=0:n
    for j=1:n_x
        Phi(i+1,j)= FEM_HatFunc(i,n,X(j));
    end
end
  

 Phi=Phi';
 figure(1);
 plot(Phi);
        
 
Phi=[]; 
x = -1:.05:2;   
for i=0:n
    Phi(i+1,:)= FEM_HatFunc(i,n,x);
end
 
 Phi=Phi';
 figure(2);
 plot(x,Phi);
    

%  plot(x,PhiPhi(x,2,3,10),'linew',4),
% Differential of hat function
Phi=[]; 
x = -0.005:.005:1.005;   
Phi= FEM_HatFunc3_Diff(1,n,x); 
 
Phi=Phi';
figure(3);
plot(x,Phi);