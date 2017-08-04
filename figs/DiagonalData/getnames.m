%% Import data from text file.
% Script for importing data from the following text file:
%
%    /home/kmbicol/Dropbox/FEniCSproj/Ndelta/tests_2D/DiagonalData/2D_adr_h1_40_delta1h_a.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2017/08/03 21:13:49

%% Initialize variables.
function names = getnames(filename)
delimiter = ',';
endRow = 1;

%% Format for each line of text:
%   column1: text (%q)
%	column2: text (%q)
%   column3: text (%q)
%	column4: text (%q)
%   column5: text (%q)
% For more information, see the TEXTSCAN documentation.
if ~isempty(findstr('a.csv',filename))
    formatSpec = '%q%q%q%q%q%[^\n\r]';
else
    formatSpec = '%q%q%q%q%q%q%q%[^\n\r]';
end

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow, 'Delimiter', delimiter, 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
names = [dataArray{1:end-1}];

%% Clear temporary variables
clearvars filename delimiter endRow formatSpec fileID dataArray ans;
end