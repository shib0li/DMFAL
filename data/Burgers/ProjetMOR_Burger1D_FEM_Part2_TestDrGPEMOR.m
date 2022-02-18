%% ProjetMOR_Burger1D_FEM_Part2_TestDrGPEMOR
%
% Modifications:
% 13-3-2016, WeiX, first edition 

clear

%%---------------Setting parameter---------------------------------------
Num_Trian=200;   % 80 is best for 'ExpData3,4.mat'
Index_test=405; % 400,403 is tracky. mostly fine. in ExpData4 404 best


n_Ubases=10;     %Number of POD bases

new_dim=5;     % For LPCA more than 10 require. The new dimension of reduced emulatior



%% DR method and parameters
dim_new=10;           
DrMethod='kPCA';
% DrMethod='DiffusionMaps';
% DrMethod='Isomaps';
% DrMethod='PCA';

switch DrMethod
    
    case 'kPCA'
        options.ker='gaussian';   
        options.new_dim=dim_new;
        options.FullRec=0;       
        options.arg=1000;   %10 wont work well,WHY? model.K is too normal distributed which leads to slow eigenvalue decay!!! So options.arg tends to be large.
        options.kAuto=1;
        
    case 'DiffusionMaps'
        options.metric ='euclidean';
        options.kernel ='gaussian'; 
        options.dim_new = dim_new;              
        options.t = 1;                     
        options.FullRec = 0;      
        % Doptions.kpara = 10000;             
        options.kAuto=1;
        options.Ztype=0;    %Output type. With/without 1st component
        
    case 'Isomaps'
        options.dim_new=dim_new;                % New dimension
        options.neighborType='k';               % Type of neighbor.Choice:1)'k';Choice:2)'epsilon'
        options.neighborPara=10;                % parameter for choosing the neighbor. number of neighbor for "k" type and radius for 'epsilon'
        options.metric='euclidean';             % Method of measurement. Metric

        %Isomap PreImage options
        preoptions.ReCoverNeighborType='k';     % Type of neighbor of new point. Choice:1)'k';Choice:2)'epsilon'
        preoptions.ReCoverNeighborPara=10;      % Parameter of neighbor of new point
        
    case 'PCA'
        options=[];
        
    otherwise 
        error('No such DR method')
end


options.DrMethod=DrMethod;
% PreImage options----------------------
preoptions.type='Exp';  %'LSE', 'Dw' OR 'Exp'
preoptions.neighbor=10;
% preoptions.type='LpcaI';
% preoptions.dim_new=10; % Use to stable the result but sacrefy accuracy
% preoptions.InMethod='ANN';


%% ----------------Load dataset-------------------------------------------
load('ExpData5.mat') %7 is ok for HH,
% Num_Trian=40;   % 80 is best for 'ExpData3,4.mat'
% Index_test=404; % 400,403 is tracky. mostly fine. in ExpData4 404 best

% X_star=[10,-10,5];
X_star=X(Index_test,:);
Y_orig=Y(Index_test,:);

Paras.Re=X(Index_test,1);
Paras.u0a=X(Index_test,2);
Paras.u0b=X(Index_test,3);

%% ---------------HDM----------------------------------------------------
[Y_Full,~,Time_HDM]=Burger1D_FEM_DBC_SolverF(Paras);
Time_HDM=Time_HDM

%% ---------------MOR Bases by perfect snapshot -----------------------------------
Y_orig_snaps=reshape(Y_orig',Paras.n+1,[]);

% n_Ubases=5;
% Y_orig_snaps=Y_orig_snaps(2:end-1,:); %take out boundary point
[U_OrigSnapS,S,~]=svd(Y_orig_snaps(2:end-1,:));  % U*S*V'=Rec_X
% [U_orig,S,~]=svd(T_orig);
U_OrigSnapS=U_OrigSnapS(:,1:n_Ubases);
eigenvalues_OrigSnapS=diag(S);


[Y_U_OrigSnapS,~,Time_ROM_U_OrigSnapS]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_OrigSnapS);

%  FiledPlot (Paras,T_orig,T_U_orig)


%% ---------------MOR Golboal Bases -----------------------------------
Y_Global_snaps=reshape(Y(1:Num_Trian,:)',Paras.n+1,[]);

% n_Ubases=5;
% Y_orig_snaps=Y_orig_snaps(2:end-1,:); %take out boundary point
[U_GlobalSnapS,S,~]=svd(Y_Global_snaps(2:end-1,:));  % U*S*V'=Rec_X
% [U_orig,S,~]=svd(T_orig);
U_GlobalSnapS=U_GlobalSnapS(:,1:n_Ubases);
eigenvalues_GlobalSnapS=diag(S);


[Y_U_GlobalSnapS,~,Time_ROM_U_GlobalSnapS]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_GlobalSnapS);

%  FiledPlot (Paras,T_orig,T_U_orig)


%% -------------MOR Bases by GPE Predicted snapshot-------------------------
% n_Ubases=5;   %number of POD basics

[Y_GPE,Time_Emu]=Func_DrGPE(X(1:Num_Trian,:),Y(1:Num_Trian,:),X_star,options,preoptions);     


%-------------------------------Rearrange---------------------------------
Y_GPESnapS=reshape(Y_GPE',Paras.n+1,[]);

% Y_star_snaps=real(Y_star_snaps);

% %-----------------Show prediction by GPR and original---------------------
% T_snaps_orig=reshape(Y_orig',Paras.Num_node_x*Paras.Num_node_y,[]);
% FiledPlot (Paras,T_snaps_orig,T_snaps_star);

%-------------------MOR FEM------------------------------------
% Y_star_snaps=Y_star_snaps(2:end-1,:); %take out boundary point
[U_GPESnapS,S,~]=svd(Y_GPESnapS(2:end-1,:));  % U*S*V'=Rec_X
U_GPESnapS=U_GPESnapS(:,1:n_Ubases);
eigenvalues_Ustar=diag(S);

[Y_U_GPESnapS,~,Time_ROM_U_GPESnapS]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_GPESnapS);
Time_ROM_U_GPESnapS;

% Make sure the index end means the same frame!!!

% figure(1)
% L1=plot(Y_orig(:),'k-');
% hold on
% L2=plot(Y_star_snaps(:),'b--');
% L3=plot(Y_U_star(:),'r-.');
% legend([L1,L2,L3],'Original Y filed','GPR Predicted Y filed','GPR-MOR Y filed')
% hold off

%% ----Meshc plot------------------------------------------------------
figure(1)
subplot(2,3,1)
meshc(Y_Full)
title(sprintf('HDM FEM Solution. Time cost= %0.3f',Time_HDM))

subplot(2,3,2)
meshc(Y_orig_snaps)
title(sprintf('HDM FEM Snapshot'))

subplot(2,3,5)
meshc(Y_GPESnapS)
title(sprintf('GPE Snapshot Solution. Time cost= %0.3f',Time_Emu))

subplot(2,3,4)
meshc(Y_U_GlobalSnapS)
title(sprintf('MOR Global bases. Time cost= %0.3f',Time_ROM_U_GlobalSnapS))


subplot(2,3,3)
meshc(Y_U_OrigSnapS)
title(sprintf('Perfect bases ROM Solution. Time cost= %0.3f',Time_ROM_U_OrigSnapS))

subplot(2,3,6)
meshc(Y_U_GPESnapS)
title(sprintf('GPE Bases ROM Solution. Time cost= %0.3f',Time_ROM_U_GPESnapS))


% ----Box plot------------------------------------------------------------
Err_Y_star_snaps=(Y_GPESnapS-Y_orig_snaps).^2;
Err_Y_U_orig=(Y_U_OrigSnapS-Y_Full).^2;
Err_Y_U_star=(Y_U_GPESnapS-Y_Full).^2;

figure(2)
boxplot([Err_Y_U_orig(:),Err_Y_U_star(:)],'labels',{'Normal MOR FEM','Emulation Bases MOR FEM'})
title(sprintf('L^2 Error on each point at each time'))

figure(3)
boxplot(Err_Y_star_snaps(:),'labels',{'Emulation Solution'}) 
title(sprintf('L^2 Error on each point at each time'))
   
% ----Animation plot------------------------------------------------------
x = 0:1/Paras.n:1;  % coordinate sequence
Y_Maxi=max([Y_Full(:);Y_U_OrigSnapS(:);Y_U_GPESnapS(:)]);
Y_Mini=min([Y_Full(:);Y_U_OrigSnapS(:);Y_U_GPESnapS(:)]);
for i = 1:Paras.t_n+1
    figure(4)   
%     title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))
    L1=plot(x,Y_Full(:,i),'k-');
    hold on
    L2=plot(x,Y_U_OrigSnapS(:,i),'b--');
    L3=plot(x,Y_U_GPESnapS(:,i),'r-.');
    L4=plot(x,Y_U_GlobalSnapS(:,i),'G--');
    
    axis([0,1,Y_Mini,Y_Maxi]);
    legend([L1,L2,L3,L4],'Full FEM Solution','Perfect MOR FEM Solution','Emulation bases MOR FEM Solution','Global bases MOR FEM Solution')
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





