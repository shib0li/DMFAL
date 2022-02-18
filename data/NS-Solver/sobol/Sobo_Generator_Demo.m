% Sobo_Generator_Demo
%
%  Modification:
%  WeiX, Nov 26th 2014, First Edition

Num=1000;
rang=[5,50;-5,-50];
    
[X] = Sobo_Generator(Num,rang);

figure(1)
scatter(X(:,1),X(:,2));

Z = exp ( (-X(:,1).^2 - X(:,2).^2)/100 );
figure(2)
scatter3(X(:,1),X(:,2),Z)



%%Demo2
%     Num=150;
%     Dim=4;
% 
%     X=i4_sobol_generate ( Dim, Num, 0 ); %0 is the seed of random
%     X=X';
% 
%     Ulid_uplimit=10;
%     Ulid_downlimit=5;
%     X(:,1)=X(:,1)*(Ulid_uplimit-Ulid_downlimit)+ones(Num,1)*Ulid_downlimit;
% 
%     Ulid_uplimit=2.5;
%     Ulid_downlimit=0;
%     X(:,2)=X(:,2)*(Ulid_uplimit-Ulid_downlimit)+ones(Num,1)*Ulid_downlimit;
% 
%     Ulid_uplimit=10;
%     Ulid_downlimit=0;
%     X(:,3)=X(:,3)*(Ulid_uplimit-Ulid_downlimit)+ones(Num,1)*Ulid_downlimit;
% 
%     Ulid_uplimit=0.9;
%     Ulid_downlimit=0;
%     X(:,4)=X(:,4)*(Ulid_uplimit-Ulid_downlimit)+ones(Num,1)*Ulid_downlimit;