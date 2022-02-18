%% ProjetMOR_Burger1D_FEM_Part2_TestDrGPEMORx3
%
% Modifications:
% 13-3-2016, WeiX, first edition 

clear

%%---------------Setting parameter---------------------------------------
Num_Trian=200;   % 80 is best for 'ExpData3,4.mat'
Index_test=326;  % ExpData8 use 428 426 as very good 326 fail


n_Ubases=15;     %Number of POD bases

dim_new=10;      % For LPCA more than 10 require. The new dimension of reduced emulatior
     


%% DR method and parameters
     
DrMethod='kPCA';
% DrMethod='DiffusionMaps';
DrMethod='Isomaps';
DrMethod='PCA';

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
load('ExpData8.mat') %7 is ok for HH, HH fails in 4.
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

% --Rearrange--
Y_GPESnapS=reshape(Y_GPE',Paras.n+1,[]);

% Y_star_snaps=real(Y_star_snaps);

% %-----------------Show prediction by GPR and original---------------------
% T_snaps_orig=reshape(Y_orig',Paras.Num_node_x*Paras.Num_node_y,[]);
% FiledPlot (Paras,T_snaps_orig,T_snaps_star);

% --MOR FEM--
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



%% -------------Figure local basis-------------------------
% Figure local basis
for i=1:Num_Trian
    
    Yi=Y(i,:);
    Yi_SnapS=reshape(Yi',Paras.n+1,[]);
    [U_YiSnapS,~,~]=svd(Yi_SnapS(2:end-1,:),'econ');  % U*S*V'=Rec_X
    U_YiSnapS=U_YiSnapS(:,1:n_Ubases);
    
    U_Y(i,:)=U_YiSnapS(:)';
    
end



%% -------------MOR Bases by GPE Prediction-------------------------
[U_GPE,Time_Emu]=Func_DrGPE(X(1:Num_Trian,:),U_Y(1:Num_Trian,:),X_star,options,preoptions);     

% --Rearrange--
U_GPE=reshape(U_GPE',Paras.n-1,[]);

[Y_U_GPE,~,Time_ROM_U_GPE]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_GPE);


%% Error Analysis

SE_U_OrigSnapS=(Y_U_OrigSnapS-Y_Full).^2;
SE_U_GPESnapS=(Y_U_GPESnapS-Y_Full).^2;
SE_U_GlobalSnapS=(Y_U_GlobalSnapS-Y_Full).^2;
SE_U_GPE=(Y_U_GPE-Y_Full).^2;

SE_Y_GPESnapS=(Y_GPESnapS-Y_orig_snaps).^2;
SE_Y_GPESnapS =repmat(SE_Y_GPESnapS,Paras.t_n/Paras.n_snap,1);
SE_Y_GPESnapS=reshape(SE_Y_GPESnapS(:),Paras.n+1,[]);

%% ----Meshc plot------------------------------------------------------
figure(1)
subplot(3,3,1)
meshc(Y_Full)
title(sprintf('HDM FEM Solution. Time cost= %0.3f',Time_HDM))

subplot(3,3,2)
meshc(Y_orig_snaps)
title(sprintf('HDM FEM Snapshot'))

subplot(3,3,5)
meshc(Y_GPESnapS)
title(sprintf('GPE Snapshot Solution. Time cost= %0.3f',Time_Emu))

subplot(3,3,4)
meshc(Y_U_GlobalSnapS)
title(sprintf('Global bases MOR Solution. Time cost= %0.3f',Time_ROM_U_GlobalSnapS))


subplot(3,3,3)
meshc(Y_U_OrigSnapS)
title(sprintf('Perfect bases ROM Solution. Time cost= %0.3f',Time_ROM_U_OrigSnapS))

subplot(3,3,6)
meshc(Y_U_GPESnapS)
title(sprintf('GPE Snapshot Bases ROM Solution. Time cost= %0.3f',Time_ROM_U_GPESnapS))

subplot(3,3,7)
meshc(Y_U_GPE)
title(sprintf('GPE Bases ROM Solution. Time cost= %0.3f',Time_ROM_U_GPE))



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


% ----L2 Error accumulation plot------------------------------------------------------------
i = 1:Paras.t_n+1;
x=(i-1)*(Paras.t_end/Paras.t_n);
figure(4)

L1=plot(x,cumsum(sum(SE_U_OrigSnapS)),'b--');
hold on
L2=plot(x,cumsum(sum(SE_U_GPESnapS)),'r-.');
L3=plot(x,cumsum(sum(SE_U_GlobalSnapS)),'g--');
L4=plot(x,cumsum(sum(SE_U_GPE)),'y--');
L5=plot(x,cumsum(sum(SE_Y_GPESnapS(:,1:Paras.t_n+1))),'m--');

legend([L1,L2,L3,L4,L5],'Perfect MOR L^2 Error','GPE Snapshot bases MOR L^2 Error','Global bases MOR L^2 Error','GPE bases MOR L^2 Error','GPE Snapshot L^2 Error')
title(sprintf('L^2 error accumulation'))
hold off


   
% ----Animation plot------------------------------------------------------
x = 0:1/Paras.n:1;  % coordinate sequence
Y_Maxi=max([Y_Full(:);Y_U_OrigSnapS(:);Y_U_GPESnapS(:)]);
Y_Mini=min([Y_Full(:);Y_U_OrigSnapS(:);Y_U_GPESnapS(:)]);
for i = 1:Paras.t_n+1
    figure(5)   
%     title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))
%     subplot(2,1,1)

    L1=plot(x,Y_Full(:,i),'k-');
    hold on
    L2=plot(x,Y_U_OrigSnapS(:,i),'b--');
    L3=plot(x,Y_U_GPESnapS(:,i),'r-.');
    L4=plot(x,Y_U_GlobalSnapS(:,i),'g--');
    L5=plot(x,Y_U_GPE(:,i),'y--');
    
    L6=plot(x,Y_GPESnapS(:,fix(i/(Paras.t_n/Paras.n_snap))+1),'m--');
    
%     if mod(i,Paras.t_n/Paras.n_snap)==1;
% %         index=fix(i/(Paras.t_n/Paras.n_snap));
%         L6=plot(x,Y_GPESnapS(:,fix(i/(Paras.t_n/Paras.n_snap))+1),'m--');
%     end
        
%     axis([0,1,Y_Mini,Y_Maxi]);
%     legend([L1,L2,L3,L4,L5,L6],'Full FEM Solution','Perfect MOR FEM Solution','GPE Snapshot bases MOR FEM Solution','Global bases MOR FEM Solution','GPE bases MOR FEM Solution','GPE prediction')
%     title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n)))) 
%     hold off
    
    
% %     figure(5)
%     subplot(2,1,2)
%     L1=plot(x(1:i),sum(Y_Full(:,1:i)-Y_Full(:,1:i)),'k-');
%     hold on
%     L2=plot(x(1:i),sum(Y_U_OrigSnapS(:,1:i)-Y_Full(:,1:i)),'b--');
%     L3=plot(x(1:i),sum(Y_U_GPESnapS(:,1:i)-Y_Full(:,1:i)),'r-.');
%     L4=plot(x(1:i),sum(Y_U_GlobalSnapS(:,1:i)-Y_Full(:,1:i)),'g--');
%     L5=plot(x(1:i),sum(Y_U_GPE(:,1:i)-Y_Full(:,1:i)),'y--');
    
    axis([0,1,Y_Mini,Y_Maxi]);
    legend([L1,L2,L3,L4,L5,L6],'Full FEM Solution','Perfect MOR FEM Solution','GPE Snapshot bases MOR FEM Solution','Global bases MOR FEM Solution','GPE bases MOR FEM Solution','GPE prediction')
    title(sprintf('L^2 Error Acumulation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n)))) 
    hold off

    F(i) = getframe;    
end    
    



% L2 Error accumulation Animation plot
% for i = 1:Paras.t_n+1
%   
%     figure(5)
% 
%     L1=plot(x(1:i),sum(Y_Full(:,1:i)-Y_Full(:,1:i)),'k-');
%     hold on
%     L2=plot(x(1:i),sum(Y_U_OrigSnapS(:,1:i)-Y_Full(:,1:i)),'b--');
%     L3=plot(x(1:i),sum(Y_U_GPESnapS(:,1:i)-Y_Full(:,1:i)),'r-.');
%     L4=plot(x(1:i),sum(Y_U_GlobalSnapS(:,1:i)-Y_Full(:,1:i)),'g--');
%     L5=plot(x(1:i),sum(Y_U_GPE(:,1:i)-Y_Full(:,1:i)),'y--');
%     
%     axis([0,1,Y_Mini,Y_Maxi]);
%     legend([L1,L2,L3,L4,L5],'Full FEM Solution','Perfect MOR FEM Solution','GPE Snapshot bases MOR FEM Solution','Global bases MOR FEM Solution','GPE bases MOR FEM Solution')
%     title(sprintf('L^2 Error Acumulation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n)))) 
%     hold off
% 
%     F(i) = getframe;    
% end    
% 
% SE_U_OrigSnapS=(Y_U_OrigSnapS-Y_Full).^2;
% SE_U_GPESnapS=(Y_U_GPESnapS-Y_Full).^2;
% SE_U_GlobalSnapS=(Y_U_GlobalSnapS-Y_Full).^2;
% SE_U_GPE=(Y_U_GPE-Y_Full).^2;







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





