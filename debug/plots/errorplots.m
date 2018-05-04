close all;

%% Evolve only, steady state (Steady ADR)
% % Exact u = iliescu without sin(pi*t)
% sigma = 1.0
% mu = 10**(-3) 
% b=as_vector([2.0, 3.0])
% nx = [25, 50, 100, 200, 400, 800]

L2 = [0.00247460360646, 0.000178438487313, 2.73358212482e-05, 3.41629105533e-06, 6.94074779253e-07, 3.6014931185e-07];

H1 = [0.306905633863, 0.0721749666942,  0.0219256336243, 0.00485695915782, 0.00092161831575, 0.000215529295391];

errplot(L2, H1, 'Evolve only, steady state, \mu =10^{-3}, P2 elements', 'k')

%%

L2a = [0.060268960767,0.0140569542371,0.00081169760276,5.74359580515e-05,1.06868210769e-05,1.6475746086e-06];

H1a = [8.20787143056,2.98967945006,0.381021312301,0.0973251772743,0.0343154282121,0.00923974756737];

errplot(L2a, H1a, 'Evolve only, steady state, \mu =10^{-4}, P2 elements', 'm')
hold on;

%%

L2b = [0.575963540234,0.263090765967,0.041851765649, 0.00554563623359,0.000221173750748,2.0995474916e-05];

H1b = [77.7640877026,66.3351391124,21.2045691927,4.4654137941,0.47857066332,0.142515173974];

errplot(L2b, H1b, 'Evolve only, steady state, \mu =10^{-5}, P2 elements', 'g')

%%

L2c = [0.368993791938, 0.365904801733, 0.364453616144,0.364351562791,0.364348676342,0.364350346201];

H1c = [10.6678477777, 11.3773983927, 7.53380792097, 4.60927008799, 4.25925064503, 4.24412160496];

errplot(L2c, H1c, 'Evolve only, time dep, \mu =10^{-5}, dt = 0.01, steady soln', 'g')

%%

L2d = [0.0239802243278,0.0239513898403,0.0237699754044,0.0237746935108,0.0237756035489,0.0237755998797];

H1d = [0.61076385581,0.749544902358,0.568590010782,0.446254919906,0.435115732696,0.434766469574];

errplot(L2c, H1c, 'Evolve only, time dep, \mu =10^{-5}, dt = 0.01, unsteady soln', 'm')

%%

L2 = [0.266828434956,0.247211208008,0.207463064338,0.152916237119,0.134728904097, 0.133319784975];

H1 = [24.5316095308,42.4136319021,62.474211178,68.8441342061,67.1809048283, 67.6577187892];

errplot(L2c, H1c, 'Evolve only, time dep, \mu =10^{-5}, dt = 1, unsteady soln', 'm')

%%

L2 = [0.00184787979202, 0.00105810225271,0.000276831534501,4.64777180384e-05,4.12684044107e-06,6.41631919236e-07];

H1 = [0.312366263529,0.332888814706,0.195881509539,0.0568651780157,0.0124115103279,0.00438898711996];


errplot(L2, H1, 'Time Discrete Problem, \mu =10^{-5}, dt = 0.01', 'm')

%%
% t =0.01
L2k = [0.00184782410145,0.0010581291447,0.000276852607002,4.6630251329e-05,5.6213014013e-06, 3.90191336512e-06];

H1k = [0.312363909881,0.332888253109,0.195881350549,0.0568651311196,0.0124115252922,0.00438917806714];

errplot(L2k, H1k, 'u_D steps forward fix, \mu =10^{-5}, t = 0.01', 'm')

% t = 0.02
L2e = [0.00477790918901,0.00273003989336,0.000694943973744,0.000114531225914,1.59420248785e-05,1.33224617959e-05];

H1e = [0.768480358603,0.833713592743, 0.478676790862,0.132601138766, 0.0258954719342,0.00885887604912];

errplot(L2e, H1e, 'u_D steps forward fix, \mu =10^{-5}, t = 0.02', 'm')

%% t = 0.5

L2g = [0.37754817099,0.19194848289,0.03698847582,0.00584696741022,0.00223471532047,0.00221318872969];

H1g = [52.1983435974,48.3205959032,18.9217034051,4.39133912945,1.1871295158,1.12508797379];

errplot(L2g, H1g, 'Evolve only, time dep, \mu =10^{-5}, t = 0.5', 'm')