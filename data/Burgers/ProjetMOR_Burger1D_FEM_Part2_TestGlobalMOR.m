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


% hold off

% ----Meshc plot------------------------------------------------------
figure(1)
subplot(2,3,1)
meshc(Y_Full)
title(sprintf('Full FEM Solution. Time cost= %0.3f',Time_HDM))

subplot(2,3,2)
meshc(Y_orig_snaps)
title(sprintf('Full FEM Snapshot Solution. Time cost= %0.3f',Time_HDM))

% subplot(2,3,5)
% meshc(Y_star_snaps)
% title(sprintf('Emulation Solution. Time cost= %0.3f',Time_Emu))

subplot(2,3,3)
meshc(Y_U_orig)
title(sprintf('Normal MOR FEM Solution. Time cost= %0.3f',Time_U_orig))

% subplot(2,3,6)
% meshc(Y_U_star)
% title(sprintf('MOR FEM emulation Bases ROM Solution. Time cost= %0.3f',Time_U_star))

   
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
    





