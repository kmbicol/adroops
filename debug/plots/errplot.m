function [ ] = errplot( L2, H1, method, color)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    L2size = size(L2,2);
    H1size = size(H1,2);
    w=1.5;
    r = 10^4; b = 10^-2;
    g = 2; d = 3;
    h2 = [r r/(2^g) r/(4^g) r/(8^g) r/(16^g) r/(32^g) r/(64^g)];
    h3 = [b b/(2^d) b/(4^d) b/(8^d) b/(16^d) b/(32^d) b/(64^d)];
    fig = figure();
    level = 0:5;
    semilogy(level,L2, [color 'o-.'],'LineWidth',w)
    hold on;
    semilogy(level,H1, [color 'x-.'],'LineWidth',w)
    hold on;
    semilogy(level(1:H1size),h3(1:H1size),'b','LineWidth',w)
    hold on;
    semilogy(level(1:L2size),h2(1:L2size),'r','LineWidth',w)

    legend('L2','H1','L2 ~ o(h^3)','H1 ~ o(h^2)','Location','southwest')
    title(['Method: ' method])
    xticks(level)
    xlabel('Refinement Level')
end


