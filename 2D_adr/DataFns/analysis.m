% goal: fix h, plot N for different delta
% cd into folder with csv data from paraview
cd ~/Dropbox/FEniCSproj/Ndelta/tests_2D/DiagonalData/

files = dir('2D_adr_h1_20*_u.csv');

a = files(1).name;
 = getnames(a);
aaa = getvalues(a);

b = files(2).name;
bb = getnames(b);
bbb = 
c = files(3).name;
cc = get names(c);
%%
for j=1:2 % fix N
    fig = figure();
    
    for i = 1:size(files,1)
        filename = files(i).name;
        names = getnames(filename); % legend
        values = getvalues(filename); % column vectors

        set(fig,'Units','normalized','Position',[0,0,1,1]);

        plot(values(:,1),values(:,j+1),'-','linewidth',1.5)
        %legend(filename(end-12:end-6),'Location','northwest','FontSize',fontsize);
        hold on
        fontsize=12;

        xlabel('arclength','FontSize',fontsize);
        set(gca,'xtick',[0:.25:1.5])
        set(gca,'FontSize',fontsize)
        %saveas(fig,[filename(1:end-4) 'm'],'png')
        %hold off
        %disp(filename);
    end
    hold off
end