%% ProjetMOR_Burger1D_FEM_Part2_TestDrGPEMORx3_MC
%  ProjetMOR_Burger1D_FEM_Part2_TestDrGPEMORx3 for multiple test cases
% Modifications:
% 13-3-2016, WeiX, first edition 

clear

%%---------------Setting parameter---------------------------------------
Num_Train=200;   % 80 is best for 'ExpData3,4.mat'
Num_Test=100;
Test_StartIndex=301;

n_Ubases=10;     %Number of POD bases

dim_new=10;      % For LPCA more than 10 require. The new dimension of reduced emulatior

%% DR method and parameters
     
DrMethod='kPCA';
% DrMethod='DiffusionMaps';
DrMethod='Isomaps';
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
load('ExpData11.mat') %7 is ok for HH, HH fails in 4.
% Num_Trian=40;   % 80 is best for 'ExpData3,4.mat'
% Index_test=404; % 400,403 is tracky. mostly fine. in ExpData4 404 best

% X_star=[10,-10,5];
X_Data=X;
Y_Data=Y;

X=X_Data(1:Num_Train,:);
Y=Y_Data(1:Num_Train,:);

X_star=X_Data(Test_StartIndex:Test_StartIndex+Num_Test-1,:);
Y_starorig=Y_Data(Test_StartIndex:Test_StartIndex+Num_Test-1,:);

% Paras.Re=X(Index_test,1);
% Paras.u0a=X(Index_test,2);
% Paras.u0b=X(Index_test,3);

%% ---------------HDM----------------------------------------------------
tic;
h = waitbar(0,'Test HDM');
for i =1:Num_Test
    Paras.Re=X_star(i,1);
    Paras.u0a=X_star(i,2);
    Paras.u0b=X_star(i,3);
    
    [Y_Full,~,~]=Burger1D_FEM_DBC_SolverF(Paras);
    Y_Full_Rec(:,:,i)=Y_Full;
    waitbar(i/Num_Test);
end
close(h);
Time_HDM=toc;


%% ---------------MOR Bases by perfect snapshot -----------------------------------
tic;
h = waitbar(0,'Test MOR Bases by perfect snapshot');
for i =1:Num_Test
    
    Paras.Re=X_star(i,1);
    Paras.u0a=X_star(i,2);
    Paras.u0b=X_star(i,3);
    Y_orig_snaps=reshape(Y_starorig(i,:)',Paras.n+1,[]);
    
    [U_OrigSnapS,~,~]=svd(Y_orig_snaps(2:end-1,:));  % U*S*V'=Rec_X
    % [U_orig,S,~]=svd(T_orig);
    U_OrigSnapS=U_OrigSnapS(:,1:n_Ubases);
%     eigenvalues_OrigSnapS=diag(S);
    [Y_U_OrigSnapS,~,~]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_OrigSnapS);
    Y_U_OrigSnapS_Rec(:,:,i)=Y_U_OrigSnapS; 
    waitbar(i/Num_Test);
end
close(h);
Time_ROM_U_OrigSnapS=toc;

%% ---------------MOR Golboal Bases -----------------------------------
Y_Global_snaps=reshape(Y(1:Num_Train,:)',Paras.n+1,[]);

% n_Ubases=5;
% Y_orig_snaps=Y_orig_snaps(2:end-1,:); %take out boundary point
[U_GlobalSnapS,~,~]=svd(Y_Global_snaps(2:end-1,:));  % U*S*V'=Rec_X
% [U_orig,S,~]=svd(T_orig);
U_GlobalSnapS=U_GlobalSnapS(:,1:n_Ubases);
% eigenvalues_GlobalSnapS=diag(S);

tic;
h = waitbar(0,'Test MOR Bases by Golboal Bases');
for i =1:Num_Test
    
    Paras.Re=X_star(i,1);
    Paras.u0a=X_star(i,2);
    Paras.u0b=X_star(i,3);
    Y_orig_snaps=reshape(Y_starorig(i,:)',Paras.n+1,[]);
    
    [Y_U_GlobalSnapS,~,Time_ROM_U_GlobalSnapS]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_GlobalSnapS);

    Y_U_GlobalSnapS_Rec(:,:,i)=Y_U_GlobalSnapS;
    waitbar(i/Num_Test);
end
close(h);
Time_ROM_U_GlobalSnapS=toc;



%  FiledPlot (Paras,T_orig,T_U_orig)


%% -------------MOR Bases by GPE Predicted snapshot-------------------------
% n_Ubases=5;   %number of POD basics

[Y_GPE,Time_Emu]=Func_DrGPE(X(1:Num_Train,:),Y(1:Num_Train,:),X_star,options,preoptions);     


tic;
h = waitbar(0,'Test MOR Bases by GPE snapshot');
for i =1:Num_Test
    
    Paras.Re=X_star(i,1);
    Paras.u0a=X_star(i,2);
    Paras.u0b=X_star(i,3);

    Y_GPESnapS=reshape(Y_GPE(i,:)',Paras.n+1,[]);
    
    [U_GPESnapS,~,~]=svd(Y_GPESnapS(2:end-1,:));  % U*S*V'=Rec_X
    U_GPESnapS=U_GPESnapS(:,1:n_Ubases);
%     eigenvalues_Ustar=diag(S);

    [Y_U_GPESnapS,~,~]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_GPESnapS);

    Y_U_GPESnapS_Rec(:,:,i)=Y_U_GPESnapS;
    waitbar(i/Num_Test);
end
close(h);
Time_ROM_U_GPESnapS=toc;


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
for i=1:Num_Train    
    Yi=Y(i,:);
    Yi_SnapS=reshape(Yi',Paras.n+1,[]);
    [U_YiSnapS,~,~]=svd(Yi_SnapS(2:end-1,:),'econ');  % U*S*V'=Rec_X
    U_YiSnapS=U_YiSnapS(:,1:n_Ubases); 
    U_Y(i,:)=U_YiSnapS(:)';
    
end

%% -------------MOR Bases by GPE Prediction-------------------------
[U_GPE,Time_Emu2]=Func_DrGPE(X(1:Num_Train,:),U_Y(1:Num_Train,:),X_star,options,preoptions);     


tic;
h = waitbar(0,'Test MOR Bases by GPE Basis');
for i =1:Num_Test
    
    Paras.Re=X_star(i,1);
    Paras.u0a=X_star(i,2);
    Paras.u0b=X_star(i,3);
    

    U_GPEi=reshape(U_GPE(i,:)',Paras.n-1,[]);
    [Y_U_GPE,~,~]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_GPEi);
    
    Y_U_GPE_Rec(:,:,i)=Y_U_GPE;
    waitbar(i/Num_Test);
end
close(h);
Time_ROM_U_GPE=toc;

% --Rearrange--

% U_GPE=reshape(U_GPE',Paras.n-1,[]);
% [Y_U_GPE,~,~]=Burger1D_FEM_DBC_MOR_SolverF(Paras,U_GPE);


%% Error Analysis
tic;
h = waitbar(0,'Error analysis');
for i =1:Num_Test
    
    SSE_dx_U_OrigSnapS(i,:)=sum((Y_U_OrigSnapS_Rec(:,:,i)-Y_Full_Rec(:,:,i)).^2,1);   %Square sum error; integral on dx
    SSE_dx_U_GlobalSnapS(i,:)=sum((Y_U_GlobalSnapS_Rec(:,:,i)-Y_Full_Rec(:,:,i)).^2,1);
    SSE_dx_U_GPESnapS(i,:) =sum((Y_U_GPESnapS_Rec(:,:,i)-Y_Full_Rec(:,:,i)).^2,1); 
    SSE_dx_Y_U_GPE(i,:)=sum((Y_U_GPE_Rec(:,:,i)-Y_Full_Rec(:,:,i)).^2,1);
    
    SSE_dxdt_U_OrigSnapS(i,:)=sum(SSE_dx_U_OrigSnapS(i,:),2);                              %Square sum error; integral on dx & dt
    SSE_dxdt_U_GlobalSnapS(i,:)=sum(SSE_dx_U_GlobalSnapS(i,:),2);                              %Square sum error; integral on dx & dt
    SSE_dxdt_U_GPESnapS(i,:)=sum(SSE_dx_U_GPESnapS(i,:),2);                              %Square sum error; integral on dx & dt
    SSE_dxdt_Y_U_GPE(i,:)=sum(SSE_dx_Y_U_GPE(i,:),2);                              %Square sum error; integral on dx & dt
    
    waitbar(i/Num_Test);
end
close(h);
toc

% SE_U_OrigSnapS=(Y_U_OrigSnapS-Y_Full).^2;
% SE_U_GPESnapS=(Y_U_GPESnapS-Y_Full).^2;
% SE_U_GlobalSnapS=(Y_U_GlobalSnapS-Y_Full).^2;
% SE_U_GPE=(Y_U_GPE-Y_Full).^2;
% 
% SE_Y_GPESnapS=(Y_GPESnapS-Y_orig_snaps).^2;
% SE_Y_GPESnapS =repmat(SE_Y_GPESnapS,Paras.t_n/Paras.n_snap,1);
% SE_Y_GPESnapS=reshape(SE_Y_GPESnapS(:),Paras.n+1,[]);

%% ---- plot------------------------------------------------------
% ----Box plot------------------------------------------------------------
% Err_Y_star_snaps=(Y_GPESnapS-Y_orig_snaps).^2;
% Err_Y_U_orig=(Y_U_OrigSnapS-Y_Full).^2;
% Err_Y_U_star=(Y_U_GPESnapS-Y_Full).^2;

figure
boxplot([SSE_dxdt_U_OrigSnapS,SSE_dxdt_U_GlobalSnapS,SSE_dxdt_U_GPESnapS,SSE_dxdt_Y_U_GPE],'labels',{'ROM ferfect base','ROM global base','ROM GPE Snapshot base','ROM GPE base'})
title(sprintf('L^2 Error'))

% figure(3)
% boxplot(Err_Y_star_snaps(:),'labels',{'Emulation Solution'}) 
% title(sprintf('L^2 Error on each point at each time'))


% % ----L2 Error accumulation plot------------------------------------------------------------
% i = 1:Paras.t_n+1;
% x=(i-1)*(Paras.t_end/Paras.t_n);
% figure(4)
% 
% L1=plot(x,cumsum(sum(SE_U_OrigSnapS)),'b--');
% hold on
% L2=plot(x,cumsum(sum(SE_U_GPESnapS)),'r-.');
% L3=plot(x,cumsum(sum(SE_U_GlobalSnapS)),'g--');
% L4=plot(x,cumsum(sum(SE_U_GPE)),'y--');
% L5=plot(x,cumsum(sum(SE_Y_GPESnapS(:,1:Paras.t_n+1))),'m--');
% 
% legend([L1,L2,L3,L4,L5],'Perfect MOR L^2 Error','GPE Snapshot bases MOR L^2 Error','Global bases MOR L^2 Error','GPE bases MOR L^2 Error','GPE Snapshot L^2 Error')
% title(sprintf('L^2 error accumulation'))
% hold off


% ----L2 Error accumulation plot Sum of all cases------------------------------------------------------------
SSE_dxdn_U_OrigSnapS=sum(sum((Y_U_OrigSnapS_Rec-Y_Full_Rec).^2,3)); %Square sum error; integral on dx & dn(cases)
SSE_dxdn_U_GlobalSnapS=sum(sum((Y_U_GlobalSnapS_Rec-Y_Full_Rec).^2,3)); %Square sum error; integral on dx & dn(cases)
SSE_dxdn_U_GPESnapS=sum(sum((Y_U_GPESnapS_Rec-Y_Full_Rec).^2,3)); %Square sum error; integral on dx & dn(cases)
SSE_dxdn_U_GPE=sum(sum((Y_U_GPE_Rec-Y_Full_Rec).^2,3)); %Square sum error; integral on dx & dn(cases)

i = 1:Paras.t_n+1;
x=(i-1)*(Paras.t_end/Paras.t_n);
figure

L1=plot(x,cumsum((SSE_dxdn_U_OrigSnapS)),'b--');
hold on
L2=plot(x,cumsum((SSE_dxdn_U_GPESnapS)),'r-.');SSE_dxdn_U_GlobalSnapS
L3=plot(x,cumsum((SSE_dxdn_U_GlobalSnapS)),'g--');
L4=plot(x,cumsum((SSE_dxdn_U_GPE)),'y--');
% L5=plot(x,cumsum(sum(SE_Y_GPESnapS(:,1:Paras.t_n+1))),'m--');

legend([L1,L2,L3,L4],'ROM ferfect base','ROM GPE Snapshot base','ROM GPE global base','ROM GPE base')
% legend([L1,L2,L3,L4,L5],'Perfect MOR L^2 Error','GPE Snapshot bases MOR L^2 Error','Global bases MOR L^2 Error','GPE bases MOR L^2 Error','GPE Snapshot L^2 Error')
title(sprintf('L^2 error accumulation'))
hold off

   
% % ----Animation plot------------------------------------------------------
% x = 0:1/Paras.n:1;  % coordinate sequence
% Y_Maxi=max([Y_Full(:);Y_U_OrigSnapS(:);Y_U_GPESnapS(:)]);
% Y_Mini=min([Y_Full(:);Y_U_OrigSnapS(:);Y_U_GPESnapS(:)]);
% for i = 1:Paras.t_n+1
%     figure(5)   
% %     title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n))))
% %     subplot(2,1,1)
% 
%     L1=plot(x,Y_Full(:,i),'k-');
%     hold on
%     L2=plot(x,Y_U_OrigSnapS(:,i),'b--');
%     L3=plot(x,Y_U_GPESnapS(:,i),'r-.');
%     L4=plot(x,Y_U_GlobalSnapS(:,i),'g--');
%     L5=plot(x,Y_U_GPE(:,i),'y--');
%     
%     if mod(i,Paras.t_n/Paras.n_snap)==1;
% %         index=fix(i/(Paras.t_n/Paras.n_snap));
%         L6=plot(x,Y_GPESnapS(:,fix(i/(Paras.t_n/Paras.n_snap))+1),'m--');
%     end
%         
% %     axis([0,1,Y_Mini,Y_Maxi]);
% %     legend([L1,L2,L3,L4,L5,L6],'Full FEM Solution','Perfect MOR FEM Solution','GPE Snapshot bases MOR FEM Solution','Global bases MOR FEM Solution','GPE bases MOR FEM Solution','GPE prediction')
% %     title(sprintf('Animation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n)))) 
% %     hold off
%     
%     
% % %     figure(5)
% %     subplot(2,1,2)
% %     L1=plot(x(1:i),sum(Y_Full(:,1:i)-Y_Full(:,1:i)),'k-');
% %     hold on
% %     L2=plot(x(1:i),sum(Y_U_OrigSnapS(:,1:i)-Y_Full(:,1:i)),'b--');
% %     L3=plot(x(1:i),sum(Y_U_GPESnapS(:,1:i)-Y_Full(:,1:i)),'r-.');
% %     L4=plot(x(1:i),sum(Y_U_GlobalSnapS(:,1:i)-Y_Full(:,1:i)),'g--');
% %     L5=plot(x(1:i),sum(Y_U_GPE(:,1:i)-Y_Full(:,1:i)),'y--');
%     
%     axis([0,1,Y_Mini,Y_Maxi]);
%     legend([L1,L2,L3,L4,L5],'Full FEM Solution','Perfect MOR FEM Solution','GPE Snapshot bases MOR FEM Solution','Global bases MOR FEM Solution','GPE bases MOR FEM Solution')
%     title(sprintf('L^2 Error Acumulation t= %0.3f',((i-1)*(Paras.t_end/Paras.t_n)))) 
%     hold off
% 
%     F(i) = getframe;    
% end    
    





