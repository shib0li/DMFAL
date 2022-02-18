%% ProjetMOR_Burger1D_FEM_Part2_TestMOR
%
% Modifications:
% 1-Jun-2015, WeiX, first edition 

clear

%%---------------Setting parameter---------------------------------------
n_Ubases=10;     %Number of POD bases

new_dim=5;     % For LPCA more than 10 require. The new dimension of reduced emulatior
%----------------Load dataset-------------------------------------------
load('ExpData4.mat') %7 is ok for HH,
Num_Trian=40;   % 80 is best for 'ExpData3,4.mat'
Index_test=404; % 400,403 is tracky. mostly fine. in ExpData4 404 best

% X_star=[10,-10,5];
X_star=X(Index_test,:);
Y_orig=Y(Index_test,:);

Paras.Re=X(Index_test,1);
Paras.u0a=X(Index_test,2);
Paras.u0b=X(Index_test,3);

% ---------------HDM----------------------------------------------------
[Y_Full,~,Time_HDM]=Burger1D_FEM_DBC_SolverF(Paras);
Time_HDM=Time_HDM

% ---------------MOR Bases by Record-----------------------------------
Y_orig_snaps=reshape(Y_orig',Paras.n+1,[]);

% n_Ubases=5;
% Y_orig_snaps=Y_orig_snaps(2:end-1,:); %take out boundary point
[U_orig,S,~]=svd(Y_orig_snaps(2:end-1,:));  % U*S*V'=Rec_X
% [U_orig,S,~]=svd(T_orig);
U_orig=U_orig(:,1:n_Ubases);
eigenvalues_Uorig=diag(S);


[Y_U_orig,~,Time_U_orig]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_orig);
Time_U_orig=Time_U_orig
%  FiledPlot (Paras,T_orig,T_U_orig)


% -------------MOR Bases by GPR------------------------------------------
% n_Ubases=5;   %number of POD basics


% -------------Making prediction using emulation--------------------------
%
% Choose one from the following method to make prediction
% ----Full-GPR------------------------------------------------------------
% [Y_star,~]= GPR_prediction(X(1:Num_Trian,:),Y(1:Num_Trian,:),X_star);
% ----LPCA-GPR------------------------------------------------------------
% [Y_star,~,Time_Emu]=GPR_SVD(X(1:Num_Trian,:),Y(1:Num_Trian,:),X_star,new_dim);

% ----KPCA-GPR------------------------------------------------------------
options = struct('ker','rbf','arg',100,'new_dim',new_dim); %100
[Y_star,~,Time_Emu]=GPR_KPCA(X(1:Num_Trian,:),Y(1:Num_Trian,:),X_star,options);

% ----ISOMAP-GPR----------------------------------------------------------
% isomap.options.dim_new=new_dim;                      % New dimension
% isomap.options.neighborType='k';               % Type of neighbor.Choice:1)'k';Choice:2)'epsilon'
% isomap.options.neighborPara=10;                % parameter for choosing the neighbor. number of neighbor for "k" type and radius for 'epsilon'
% isomap.options.metric='euclidean';             % Method of measurement. Metric
% 
% % Isomap PreImage options
% isomap.Reoptions.ReCoverNeighborType='k';      % Type of neighbor of new point. Choice:1)'k';Choice:2)'epsilon'
% isomap.Reoptions.ReCoverNeighborPara=10;       % Parameter of neighbor of new point
% isomap.Reoptions.Recoverd2pType='Dw';          % Type of distance to coordinate method. Distance weight/Least square estimate. alternative:LSE
% isomap.Reoptions.Recoverd2pPara=2;             % Parameter of distance to coordinate recover method
% 
% [Y_star,~,Time_Emu]=GPR_Isomap2(X(1:Num_Trian,:),Y(1:Num_Trian,:),X_star,isomap.options,isomap.Reoptions);

% ----LPCA-ANN------------------------------------------------------------
% [Y_star,Time_Emu]=ANN_SVD(X(1:Num_Trian,:)',Y(1:Num_Trian,:)',X_star',new_dim,10); %works fine
% Y_star=Y_star';

% ----KPCA-ANN------------------------------------------------------------
% options = struct('ker','rbf','arg',100,'new_dim',new_dim); 
% [Y_star,Time_Emu]=ANN_KPCA(X(1:Num_Trian,:)',Y(1:Num_Trian,:)',X_star',options,10);
% Y_star=Y_star';

%-------------------------------Rearrange---------------------------------
Y_star_snaps=reshape(Y_star',Paras.n+1,[]);

% Y_star_snaps=real(Y_star_snaps);

% %-----------------Show prediction by GPR and original---------------------
% T_snaps_orig=reshape(Y_orig',Paras.Num_node_x*Paras.Num_node_y,[]);
% FiledPlot (Paras,T_snaps_orig,T_snaps_star);

%-------------------MOR FEM------------------------------------
% Y_star_snaps=Y_star_snaps(2:end-1,:); %take out boundary point
[U_star,S,~]=svd(Y_star_snaps(2:end-1,:));  % U*S*V'=Rec_X
U_star=U_star(:,1:n_Ubases);
eigenvalues_Ustar=diag(S);

[Y_U_star,~,Time_U_star]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_star);
Time_U_star;

% Make sure the index end means the same frame!!!

% figure(1)
% L1=plot(Y_orig(:),'k-');
% hold on
% L2=plot(Y_star_snaps(:),'b--');
% L3=plot(Y_U_star(:),'r-.');
% legend([L1,L2,L3],'Original Y filed','GPR Predicted Y filed','GPR-MOR Y filed')
% hold off

% ----Meshc plot------------------------------------------------------
figure(1)
subplot(2,3,1)
meshc(Y_Full)
title(sprintf('Full FEM Solution. Time cost= %0.3f',Time_HDM))

subplot(2,3,2)
meshc(Y_orig_snaps)
title(sprintf('Full FEM Snapshot Solution. Time cost= %0.3f',Time_HDM))

subplot(2,3,5)
meshc(Y_star_snaps)
title(sprintf('Emulation Solution. Time cost= %0.3f',Time_Emu))

subplot(2,3,3)
meshc(Y_U_orig)
title(sprintf('Normal MOR FEM Solution. Time cost= %0.3f',Time_U_orig))

subplot(2,3,6)
meshc(Y_U_star)
title(sprintf('MOR FEM emulation Bases ROM Solution. Time cost= %0.3f',Time_U_star))


% ----Box plot------------------------------------------------------------
Err_Y_star_snaps=(Y_star_snaps-Y_orig_snaps).^2;
Err_Y_U_orig=(Y_U_orig-Y_Full).^2;
Err_Y_U_star=(Y_U_star-Y_Full).^2;

figure(2)
boxplot([Err_Y_U_orig(:),Err_Y_U_star(:)],'labels',{'Normal MOR FEM','Emulation Bases MOR FEM'})
title(sprintf('L^2 Error on each point at each time'))

figure(3)
boxplot(Err_Y_star_snaps(:),'labels',{'Emulation Solution'}) 
title(sprintf('L^2 Error on each point at each time'))
   
% ----Animation plot------------------------------------------------------
x = 0:1/Paras.n:1;  % coordinate sequence
Y_Maxi=max([Y_Full(:);Y_U_orig(:);Y_U_star(:)]);
Y_Mini=min([Y_Full(:);Y_U_orig(:);Y_U_star(:)]);
for i = 1:Paras.t_n+1
    figure(4)   
%     title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))
    L1=plot(x,Y_Full(:,i),'k-');
    hold on
    L2=plot(x,Y_U_orig(:,i),'b--');
    L3=plot(x,Y_U_star(:,i),'r-.');
    axis([0,1,Y_Mini,Y_Maxi]);
    legend([L1,L2,L3],'Full FEM Solution','Normal MOR FEM Solution','Emulation bases MOR FEM Solution')
    title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))
    
    hold off
    F(i) = getframe;    
end    
    

% FiledPlot (Paras,T_orig,T_U_orig,T_U_star)

%% Additional try % not working

% T_Ref=T_snaps_star;
% 
% Plotline=[];
% for i =1:10
%    
%     [U_star,S,~]=svd(T_Ref);  % U*S*V'=Rec_X
%     U_star=U_star(:,1:n_Ubases);
%     eigenvalues=diag(S);
% 
%     [T_U_star,Time_U_star]=Us_HeatDC_2D_FVM_Mor(Paras,U_star);
%     Time_U_star;
%     
%        
% %     L1=plot(T_Ref(:,end),'k-');
%     
%     T_Ref=T_U_star;
%     Plotline=[Plotline,T_U_star(:,end)];
% end
% 
% figure()
% hold on
% L1=plot(Plotline);
% L1=plot(T_orig(:,end),'k--');





