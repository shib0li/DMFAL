clc;clear all;close all;
nsample=1000;
nelx=50;nely=50;volfrac=0.4;penal=3;rmin=1.5;
infl=(rand(1,nsample))*25+0.5;
thetaa=rand(1,nsample)*(pi/2);
par=[floor(infl);thetaa];
%load par.mat;
xL=[];xH=[];
for i=1:nsample
    mag=1;theta=par(2,i);Ey=1;volfrac=0.4;rmin=1.5;eno=par(1,i)*2;
    tic;
    [sol_L,obj_L,itr_L]=LBracket_generator(nelx,nely,volfrac,penal,1.5,theta,eno);
    timeL(i)=toc;
    tic;
    [sol_H,obj_H,itr_H]=LBracket_generator(nelx*4,nely*4,volfrac,penal,2,theta,eno*4);
    timeH(i)=toc;
    compH(i)=obj_H;compL(i)=obj_L;
    xsolH(:,:,i)=sol_H;xsolL(:,:,i)=sol_L;
    iterationsH(i)=itr_H;iterationsL(i)=itr_L;
    i
    %save data_LBracket_200 xsolL xsolH compL compH iterationsL iterationsH timeL timeH; 
end

smpls=randi([1,1000],1, 10);
for j=1:10
i=smpls(j);
xh=xsolH(:,:,i);xl=xsolL(:,:,i);
pause(0.1);
subplot(1,2,1);
colormap(gray); imagesc(1-xl); caxis([0 1]); axis equal; axis off; drawnow;
subplot(1,2,2);
colormap(gray); imagesc(1-xh); caxis([0 1]); axis equal; axis off; drawnow;
end;

