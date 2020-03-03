% cd into folder with csv data from paraview
cd ~/Dropbox/FEniCSproj/Ndelta/tests_2D/DataFns/

M = 1; % M=1 across delta to plot; for across h else, M=0

files = dir('*.csv');
for i=1:size(files,1)
    filename = files(i).name;
    names = getnames(filename); % legend
    values = getvalues(filename); % column vectors
    fig = figure();
    %set(fig,'Units','normalized','Position',[0,0,1,1]);
    
    for j=1:size(names,2)-1
        plot(values(:,1),values(:,j+1),'-','linewidth',1.5)
        hold on
    end
    fontsize=12;
    legend(names(2:end-M),'Location','northwest','FontSize',fontsize);
    xlabel('arclength','FontSize',fontsize);
    set(gca,'xtick',[0:.25:1.5])
    set(gca,'FontSize',fontsize)
    saveas(fig,[filename(1:end-4) 'm'],'png')
    hold off
    disp(filename);
end



%close all
 