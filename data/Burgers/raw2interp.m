% raw2Uniform_interp
% raw data interpolation to values at fix locations.
clear 

n_level = 3;

% space_interp = linspace(0,1,26);
% time_interp = linspace(0,3,26);
% space_interp = linspace(0,1,64);
% time_interp = linspace(0,3,11);
% time_interp = 1.5;

space_interp = linspace(0,1,100);
% time_interp = linspace(0,3,11);
time_interp = linspace(0,3,100);

[G_time_interp,G_space_interp] = meshgrid(time_interp,space_interp);

for i = 1:n_level
    
    raw_data_name = ['level',num2str(i),'raw'];
    load(raw_data_name);
    
    for j = 1:size(Rec_Y,1)
        [G_time_raw,G_space_raw] = meshgrid(time,space);   
        interp_Y{i}(j,:,:) = interp2(G_time_raw,G_space_raw,squeeze(Rec_Y(j,:,:)),G_time_interp,G_space_interp);
    end
 
end
%% save 
Y_lv1 = interp_Y{1};
Y_lv2 = interp_Y{2};
Y_lv3 = interp_Y{3};
X = re_List;

% save('Uniform_interp','Y_lv1','Y_lv2','Y_lv3','X')

%% show different levels
show_id = 5;
for i = 1:length(time_interp)   
    figure(1)
    clf
    hold on 
    plot(space_interp,interp_Y{1}(show_id,:,i))
    plot(space_interp,interp_Y{2}(show_id,:,i))
    plot(space_interp,interp_Y{3}(show_id,:,i))
    hold off
    
    axis([0,1,-0.2,1]);
    title(sprintf('Animation t= %0.3f',time_interp(i)))
    pause(0.2);  
end