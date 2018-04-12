function [ ] = errplot( L2, H1, method, color)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    L2size = size(L2,2);
    H1size = size(H1,2);
    w=1.5;
    r = 10^4; b = 10^-2;
    h2 = [r r/(2^2) r/(4^2) r/(8^2) r/(16^2) r/(32^2) r/(64^2)];
    h3 = [b b/(2^3) b/(4^3) b/(8^3) b/(16^3) b/(32^3) b/(64^3)];
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


