methodnum = 2;

if methodnum == 1
    method = 'No Stabilization';
    L2 = [0.7541990401	0.2252137342	0.07348682714	0.01122839499 0.0005628491421];
    H1 = [201.418771	127.0534136	73.37112525	18.5808289 1.53744538];

%% EFR
elseif methodnum == 2
    method = 'EFR';
    % solution-dependent a
    L2 = [0.02428183473	0.05229343329	0.01274862624	0.006162226424	0.002783612054];
    H1 = [0.5752817122	1.271628408	1.097009735	1.452826734	1.547680874];
    
    % a = 1 everywhere
    L2a = [0.02499418511	0.02871716935	0.02533699828	0.01098310928	0.004876786189	0.002159554714];
    H1a = [0.1226057036	0.2155217464	0.2698942904	0.2265889704	0.1907616427	0.1612833137];
%% SUPG
else
    method = 'SUPG';
    L2 = [0.170327478	0.03773574634	0.01542754342	0.003898711935 0.0003462879615];
    H1 = [14.16437567	7.344249233	4.966080195	2.062446726 0.3406369631];
end
%% Plots
w=1.5;
r = 10^4; b = 10^-2;
h2 = [r r/2^2 r/4^2 r/8^2 r/16^2 r/32^2];
h3 = [b b/2^3 b/4^3 b/8^3 b/16^3 b/32^3];
fig = figure(methodnum+1);
level = 0:1:5;
semilogy(level,L2a,'ko-.','LineWidth',w)
hold on;
semilogy(level,H1a,'kx-.','LineWidth',w)
hold on;
semilogy(level(1:5),L2,'bo-.','LineWidth',w)
hold on;
semilogy(level(1:5),H1,'rx-.','LineWidth',w)
hold on;
semilogy(level,h3,'b','LineWidth',w)
hold on;
semilogy(level,h2,'r','LineWidth',w)

legend('a=1, L2','a=1, H1','a(v), L2','a(v), H1','o(h^3)','o(h^2)','Location','southwest')
title(['Method: ' method])
xticks(level)
xlabel('Refinement Level')
pause(2.0)
saveas(fig,[method '_err_a1.png']);
