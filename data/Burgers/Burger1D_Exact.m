function y = Burger1D_Exact( t,x,n_term,v )
%Burgers equation 1D exact solution 
% input         n_term  number of the term in expansion 
%               v       viscosity
%               t       time        t>=0;
%               x       coordinate  0<=x<=1;

global q k      % q refer to viscosity here
q=v;

a0 = integral(@(x)a0_func(x),0,1);
for k=1:n_term
    a(k) = 2*integral(@(x)an_func(x),0,1);
end

sum1=0;
sum2=0;

for k=1:n_term
    sum1=sum1+(a(k)*(exp(-k^2*pi^2*q*t)*k*sin(k*pi*x)));
    sum2=sum2+(a(k)*(exp(-k^2*pi^2*q*t)*cos(k*pi*x)));
end

y=2*pi*q*sum1/(a0+sum2);


end


function y=a0_func(x)
    global q 
    y=exp(-((2*pi*q)^(-1))*(1-cos(pi*x)));
end

function y=an_func(x)
    global q k
    y=(exp(-((2*pi*q)^(-1))*(1-cos(pi*x))).*cos(k*pi*x));
end