close all;

plot = 0;
if plot == 1 %dx
    level = [1 2 3 4 5];
else %dt
    level = [0 1 2];
end

for methodnum = 0:1:2

    %% No Stabilization
    if plot == 1
        if methodnum == 1
        method = 'No Stabilization';
        L2 = [0.7541990401	0.2252137342	0.07348682714	0.01122839499 0.0005628491421];
        H1 = [201.418771	127.0534136	73.37112525	18.5808289 1.53744538];

        %% EFR
        elseif methodnum == 2
        method = 'EFR';
        L2 = [0.05229343329	0.01274862624	0.006162226424	0.002783612054 0];
        H1 = [1.271628408	1.097009735	1.452826734	1.547680874 0];
        %     
        %% SUPG
        else
        method = 'SUPG';
        L2 = [0.170327478	0.03773574634	0.01542754342	0.003898711935 0.0003462879615];
        H1 = [14.16437567	7.344249233	4.966080195	2.062446726 0.3406369631];
        end
        %% Plots
        w=1.5;
        r = 10^4; b = 10^-2;
        h2 = [r r/2^2 r/4^2 r/8^2 r/16^2];
        h3 = [b b/2^3 b/4^3 b/8^3 b/16^3];
        fig = figure(methodnum+1);
        semilogy(level,L2,'bo-.','LineWidth',w)
        hold on;
        semilogy(level,H1,'rx-.','LineWidth',w)
        hold on;
        semilogy(level,h3,'b','LineWidth',w)
        hold on;
        semilogy(level,h2,'r','LineWidth',w)

        legend('L2-norm','H1-norm','O(h^3)','O(h^2)','Location','southwest')
        title(['Method: ' method])
        xticks(level)
        xlabel('Refinement Level')
        pause(2.0)
        saveas(fig,[method '_err.png']);
    else
        if methodnum == 1
        method = 'No Stabilization';
        L2 = [0.09826398652	0.09845568931	0.09856508733];
        H1 = [104.1934564	104.3929598	104.5032794];

        %% EFR
        elseif methodnum == 2
        method = 'EFR';
        L2 = [0.03287729823	0.02950592922	0.02763063282];
        H1 = [12.71487971	10.90236518	9.559298171];
        %     
        %% SUPG
        else
        method = 'SUPG';
        L2 = [0.05089949161	0.05094556443	0.05097998632];
        H1 = [20.93962419	20.96319386	20.97581596];
        end

        %% Plots
        w=1.5;
        r = 10^4; b = 10^-2;
        h2 = [r r/2^2 r/4^2];% r/8^2 r/16^2];
        h3 = [b b/2^3 b/4^3];% b/8^3 b/16^3];
        fig = figure(methodnum+1);
        semilogy(level,L2,'bo-.','LineWidth',w)
        hold on;
        semilogy(level,H1,'rx-.','LineWidth',w)
        hold on;
        semilogy(level,h3,'b','LineWidth',w)
        hold on;
        semilogy(level,h2,'r','LineWidth',w)

        legend('L2-norm','H1-norm','O(h^3)','O(h^2)','Location','southwest')
        title(['Method: ' method])
        xticks(level)
        xlabel('Refinement Level')
        pause(2.0)
        saveas(fig,[method '_err_t0.5.png']);
    end
end