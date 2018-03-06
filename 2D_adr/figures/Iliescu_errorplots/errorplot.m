close all;


for methodnum = 0:1:2

%% No Stabilization
    if methodnum == 1
        method = 'No Stabilization';
        L2 = [0.7541990401	0.2252137342	0.07348682714	0.01122839499];
        H1 = [201.418771	127.0534136	73.37112525	18.5808289];

    %% EFR
    elseif methodnum == 2
        method = 'EFR';
        L2 = [0.05229343329	0.01274862624	0.006162226424	0.002783612054];
        H1 = [1.271628408	1.097009735	1.452826734	1.547680874];
    %     
    %% SUPG
    else
        method = 'SUPG';
        L2 = [0.170327478	0.03773574634	0.01542754342	0.003898711935];
        H1 = [14.16437567	7.344249233	4.966080195	2.062446726];
    end
    %% Plots

    level = [1 2 3 4];
    w=1.5;
    r = 10^4; b = 10^-2;
    h2 = [r r/2^2 r/4^2 r/8^2];
    h3 = [b b/2^3 b/4^3 b/8^3];
    fig = figure(methodnum+1);
    semilogy(level,L2,'bo-.','LineWidth',w)
    hold on;
    semilogy(level,H1,'rx-.','LineWidth',w)
    hold on;
    semilogy([1 2 3 4],h3,'b','LineWidth',w)
    hold on;
    semilogy([1 2 3 4],h2,'r','LineWidth',w)
    
    legend('L2-norm','H1-norm','O(h^3)','O(h^2)','Location','southwest')
    title(['Method: ' method])
    xticks([0 1 2 3 4])
    xlabel('Refinement Level')
    pause(2.0)
    saveas(fig,[method '_err.png']);
end