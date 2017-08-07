% cd into folder with csv data from paraview

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
    legend(names(2:end),'Location','northwest','FontSize',fontsize);
    xlabel('arclength','FontSize',fontsize);
    set(gca,'xtick',[0:.25:1.5])
    set(gca,'FontSize',fontsize)
    saveas(fig,[filename(1:end-4) 'm'],'epsc')
    hold off
    
end
close all
 