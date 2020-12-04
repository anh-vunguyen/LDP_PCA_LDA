%% Auteur : NGUYEN Anh Vu
%% Instructeur :  Oscar ACOSTA
%% FACE RECOGNITION WITH THE FRAMEWORK OF SUBSPACE LEARNING BASED ON LOCAL TEXTURE PATTERNS
%% Reference: Fujin Zhong, Shuo Yan, Li Liu and Ke Liu, “An Effective Face Recognition Framework With Subspace Learning Based on Local Texture Patterns,” 
%% 2018 5th International Conference on Systems and Informatics (ICSAI) Systems and Informatics (ICSAI), 2018 5th International Conference on., Nov, 2018, IEEE 2018.

%% Load the Database of Images
load('ImDatabase.mat');

%% Useful coefficients
% Number of class
NoC = 30;
% Number of images per class
NoI = 15;
% Dimension of each image
dimImg = 32256; 
% 50 largest eigen vectors
nb = 50;

%% Kirsch Operator
KirschE = [-3 -3 5; -3 0 5; -3 -3 5]; %East
KirschNE = [-3 5 5; -3 0 5; -3 -3 -3]; %NorthEast
KirschN = [5 5 5; -3 0 -3; -3 -3 -3]; %North
KirschNW = [5 5 -3; 5 0 -3; -3 -3 -3]; %NorthWest
KirschW= [5 -3 -3; 5 0 -3; 5 -3 -3]; %West
KirschSW= [-3 -3 -3; 5 0 -3; 5 5 -3]; %SouthWest
KirschS= [-3 -3 -3; -3 0 -3; 5 5 5]; %South
KirschSE= [-3 -3 -3; -3 0 5; -3 5 5]; %SouthEast

%% Kirsch Masked Matrix Initilization
KM = zeros(192,168);

%% Matrix of concatenated histogram Initilization
MatrixH = zeros(256*49,NoC*NoI);


for iS = 1:1:NoC*NoI


    Temp = 255*reshape(A(1:192*168,iS),192,168);
    for i = 2:1:191
        for j = 2:1:167
            N = KirschN(1,1)*Temp(i-1,j-1) + KirschN(1,2)*Temp(i-1,j) + KirschN(1,3)*Temp(i-1,j+1) + KirschN(2,3)*Temp(i,j+1) + KirschN(3,3)*Temp(i+1,j+1) + KirschN(3,2)*Temp(i+1,j) + KirschN(3,1)*Temp(i+1,j-1) + KirschN(2,1)*Temp(i,j-1);
            NE = KirschNE(1,1)*Temp(i-1,j-1) + KirschNE(1,2)*Temp(i-1,j) + KirschNE(1,3)*Temp(i-1,j+1) + KirschNE(2,3)*Temp(i,j+1) + KirschNE(3,3)*Temp(i+1,j+1) + KirschNE(3,2)*Temp(i+1,j) + KirschNE(3,1)*Temp(i+1,j-1) + KirschNE(2,1)*Temp(i,j-1);
            E = KirschE(1,1)*Temp(i-1,j-1) + KirschE(1,2)*Temp(i-1,j) + KirschE(1,3)*Temp(i-1,j+1) + KirschE(2,3)*Temp(i,j+1) + KirschE(3,3)*Temp(i+1,j+1) + KirschE(3,2)*Temp(i+1,j) + KirschE(3,1)*Temp(i+1,j-1) + KirschE(2,1)*Temp(i,j-1);
            SE = KirschSE(1,1)*Temp(i-1,j-1) + KirschSE(1,2)*Temp(i-1,j) + KirschSE(1,3)*Temp(i-1,j+1) + KirschSE(2,3)*Temp(i,j+1) + KirschSE(3,3)*Temp(i+1,j+1) + KirschSE(3,2)*Temp(i+1,j) + KirschSE(3,1)*Temp(i+1,j-1) + KirschSE(2,1)*Temp(i,j-1);
            S = KirschS(1,1)*Temp(i-1,j-1) + KirschS(1,2)*Temp(i-1,j) + KirschS(1,3)*Temp(i-1,j+1) + KirschS(2,3)*Temp(i,j+1) + KirschS(3,3)*Temp(i+1,j+1) + KirschS(3,2)*Temp(i+1,j) + KirschS(3,1)*Temp(i+1,j-1) + KirschS(2,1)*Temp(i,j-1);
            SW = KirschSW(1,1)*Temp(i-1,j-1) + KirschSW(1,2)*Temp(i-1,j) + KirschSW(1,3)*Temp(i-1,j+1) + KirschSW(2,3)*Temp(i,j+1) + KirschSW(3,3)*Temp(i+1,j+1) + KirschSW(3,2)*Temp(i+1,j) + KirschSW(3,1)*Temp(i+1,j-1) + KirschSW(2,1)*Temp(i,j-1);
            W = KirschW(1,1)*Temp(i-1,j-1) + KirschW(1,2)*Temp(i-1,j) + KirschW(1,3)*Temp(i-1,j+1) + KirschW(2,3)*Temp(i,j+1) + KirschW(3,3)*Temp(i+1,j+1) + KirschW(3,2)*Temp(i+1,j) + KirschW(3,1)*Temp(i+1,j-1) + KirschW(2,1)*Temp(i,j-1);
            NW = KirschNW(1,1)*Temp(i-1,j-1) + KirschNW(1,2)*Temp(i-1,j) + KirschNW(1,3)*Temp(i-1,j+1) + KirschNW(2,3)*Temp(i,j+1) + KirschNW(3,3)*Temp(i+1,j+1) + KirschNW(3,2)*Temp(i+1,j) + KirschNW(3,1)*Temp(i+1,j-1) + KirschNW(2,1)*Temp(i,j-1);

            % TEST RESULT MATRIX
            % TR = [NW N NE; W 0 E; SW S SE]

            % ARRAY OF RESULT OF KIRSCH MASK
            Kir_Org = [NW N NE E SE S SW W];
            Kir_Sorted = sort(Kir_Org);

            for ki = 1:1:8
                if Kir_Org(1,ki) >= Kir_Sorted(1,4)
                    Kir_Org(1,ki) = 1;
                else
                    Kir_Org(1,ki) = 0;
                end
            end
            KM(i,j) = 128*Kir_Org(1,1) + 64*Kir_Org(1,2) + 32*Kir_Org(1,3) + 16*Kir_Org(1,4) + 8*Kir_Org(1,5) + 4*Kir_Org(1,6) + 2*Kir_Org(1,7) + Kir_Org(1,8);
        end
    end

    % Particular Columns/Rows/Cells
    KM(1,1) = Temp(1,1);
    KM(1,168) = Temp(1,168);
    KM(192,1) = Temp(192,1);
    KM(192,168) = Temp(192,168);

    KM(1,2:167) = Temp(1,2:167);
    KM(2:191,168) = Temp(2:191,168);
    KM(192,2:167) = Temp(192,2:167);
    KM(2:191,1) = Temp(2:191,1);

    % % Before and After Kirsch Mask
    %  figure (1)
    %  imshow(Temp/255);
    
    %  figure (2)
    %  imshow(KM/255);

    % Divide into 7*7 blocks, each block 189x168 px
    KM = KM(2:190,:);

    % Matrix of block of images
    blockedM = zeros(27*49,24);
    iM = 1;   

    % Matrix of concatenated histograms 
    for i=1:1:7
        for j=1:1:7
            blockedM((iM-1)*27+1:iM*27,1:24) = KM((i-1)*27+1:i*27,(j-1)*24+1:j*24); 
            iM = iM + 1;
        end
    end
    
    % Test Show Histogram
    % imhist((blockedM((i-1)*27+1:i*27,1:24))/255);

    for i=1:1:49
        MatrixH((i-1)*256+1:i*256,iS) = imhist(blockedM((i-1)*27+1:i*27,1:24)/255);    
    end

    %Show Histogram
    % figure (3)
    % plot(MatrixH(:,1));

    % Testing 
    % figure (3)
    % imshow((blockedM(28:54,1:24)- KM(1:27,25:48))/255);
  
end

save('MatrixHistogram.mat','MatrixH');
% If the matrix of concatenated histograms MatrixH existed, load MatrixH
load('MatrixHistogram.mat');

% The mean of all concatenated histogram
meanH = mean(MatrixH,2);
save('MeanHistogram.mat','meanH');
% If the mean of all concatenated histogram meanH existed, load meanH
load('MeanHistogram.mat');

% The total scatter matrix St
% Run to make the total scatter matrix St
St = 0;
for i=1:1:NoC*NoI
    St = (MatrixH(:,i)-meanH)*(MatrixH(:,i)-meanH)';
end
save('ScatterMatrix.mat','St')

% If the total scatter matrix St existed, load the scatter matrix
load('ScatterMatrix.mat');

% The optimal projection PCA matrix Wpca
[Vpca, Dpca] = eig(St);
save('EigValuePCA.mat','Dpca');
save('EigVectorPCA.mat','Vpca');

% If EigValuePCA.mat and EigVectorPCA.mat existed, load Dpca and Vpca
load('EigValuePCA.mat');
load('EigVectorPCA.mat');

ss = diag(Dpca);
[ss,iii] = sort(-ss);
Vpca = Vpca(:,iii);
Wpca=Vpca(:,1:nb);
save('MatrixPCA.mat','Wpca');

% If the optimal projection PCA matrix Wpca existed, load Wpca
load('MatrixPCA.mat');
% Mean histogram of each class
MHeC = zeros(49*256,NoC); % 50 largest eigen vectors
for i = 1:1:NoC
    MHeC(:,i) = mean(MatrixH(:,NoI*(i-1)+1:NoI*i),2);
end

% Between-class scatter matrix Sb
Sb  = zeros(49*256,49*256);
for i=1:1:NoC
    Sb = Sb + NoI*(MHeC(:,i)-meanH)*(MHeC(:,i)-meanH)';
end

save('Sb.mat','Sb');

% If the between-class scatter matrix Sb existed, load Sb
load('Sb.mat');

% Within-class scatter matrix Sw
Sw  = zeros(49*256,49*256);
for i=1:1:NoC
    for j=1:1:NoI
        Sw = Sw + (MatrixH(:,(i-1)*NoI + j)-MHeC(:,i))*(MatrixH(:,(i-1)*NoI + j)-MHeC(:,i))';
    end
end

save('Sw.mat','Sw');

% If the within-class scatter matrix Sw existed, load Sw
load('Sw.mat');

% The optimal projection LDA matrix Wlda

% Sb_pca - Sb projected by PCA
Sb_pca = Wpca'*Sb*Wpca;
% Sw_pca - Sw projected by PCA
Sw_pca = Wpca'*Sw*Wpca;

% Find Optimal Projection Matrix
[Wlda, EVlda, Dlda] = eig(Sb_pca,Sw_pca);

save('Wlda.mat','Wlda');

% If the optimal projection LDA matrix Wlda existed, load Wlda
load('Wlda.mat');
% The optimal discriminant projection matrix Wopt
Wopt = Wpca*Wlda;
save('Wopt.mat','Wopt');

% If the optimal discriminant projection matrix Wopt existed, load Wopt
load('Wopt.mat');
%% The low-dimensional disciminant features of all training samples
% MatrixHProj = Wopt'*MatrixH;
% save('MatrixHProj.mat','MatrixHProj');

% If the low-dimensional disciminant features of all training samples  MatrixHProj existed, load MatrixHProj
load('MatrixHProj.mat');

%% Test image histogram
%Input Face
input = 'yaleB01_P00A-035E+65.pgm';
inputImg_8 = imread(input);
inputImg_db = im2double(inputImg_8);
figure ('name', 'Test Image')
imshow(inputImg_db);

%Kirsch Masked Test Matrix
KM = zeros(192,168);

% Matrix of concatenated histogram:
HisTest = zeros(256*49,1);
Temp = 255*reshape(inputImg_db,192,168);
for i = 2:1:191
    for j = 2:1:167
        N = KirschN(1,1)*Temp(i-1,j-1) + KirschN(1,2)*Temp(i-1,j) + KirschN(1,3)*Temp(i-1,j+1) + KirschN(2,3)*Temp(i,j+1) + KirschN(3,3)*Temp(i+1,j+1) + KirschN(3,2)*Temp(i+1,j) + KirschN(3,1)*Temp(i+1,j-1) + KirschN(2,1)*Temp(i,j-1);
        NE = KirschNE(1,1)*Temp(i-1,j-1) + KirschNE(1,2)*Temp(i-1,j) + KirschNE(1,3)*Temp(i-1,j+1) + KirschNE(2,3)*Temp(i,j+1) + KirschNE(3,3)*Temp(i+1,j+1) + KirschNE(3,2)*Temp(i+1,j) + KirschNE(3,1)*Temp(i+1,j-1) + KirschNE(2,1)*Temp(i,j-1);
        E = KirschE(1,1)*Temp(i-1,j-1) + KirschE(1,2)*Temp(i-1,j) + KirschE(1,3)*Temp(i-1,j+1) + KirschE(2,3)*Temp(i,j+1) + KirschE(3,3)*Temp(i+1,j+1) + KirschE(3,2)*Temp(i+1,j) + KirschE(3,1)*Temp(i+1,j-1) + KirschE(2,1)*Temp(i,j-1);
        SE = KirschSE(1,1)*Temp(i-1,j-1) + KirschSE(1,2)*Temp(i-1,j) + KirschSE(1,3)*Temp(i-1,j+1) + KirschSE(2,3)*Temp(i,j+1) + KirschSE(3,3)*Temp(i+1,j+1) + KirschSE(3,2)*Temp(i+1,j) + KirschSE(3,1)*Temp(i+1,j-1) + KirschSE(2,1)*Temp(i,j-1);
        S = KirschS(1,1)*Temp(i-1,j-1) + KirschS(1,2)*Temp(i-1,j) + KirschS(1,3)*Temp(i-1,j+1) + KirschS(2,3)*Temp(i,j+1) + KirschS(3,3)*Temp(i+1,j+1) + KirschS(3,2)*Temp(i+1,j) + KirschS(3,1)*Temp(i+1,j-1) + KirschS(2,1)*Temp(i,j-1);
        SW = KirschSW(1,1)*Temp(i-1,j-1) + KirschSW(1,2)*Temp(i-1,j) + KirschSW(1,3)*Temp(i-1,j+1) + KirschSW(2,3)*Temp(i,j+1) + KirschSW(3,3)*Temp(i+1,j+1) + KirschSW(3,2)*Temp(i+1,j) + KirschSW(3,1)*Temp(i+1,j-1) + KirschSW(2,1)*Temp(i,j-1);
        W = KirschW(1,1)*Temp(i-1,j-1) + KirschW(1,2)*Temp(i-1,j) + KirschW(1,3)*Temp(i-1,j+1) + KirschW(2,3)*Temp(i,j+1) + KirschW(3,3)*Temp(i+1,j+1) + KirschW(3,2)*Temp(i+1,j) + KirschW(3,1)*Temp(i+1,j-1) + KirschW(2,1)*Temp(i,j-1);
        NW = KirschNW(1,1)*Temp(i-1,j-1) + KirschNW(1,2)*Temp(i-1,j) + KirschNW(1,3)*Temp(i-1,j+1) + KirschNW(2,3)*Temp(i,j+1) + KirschNW(3,3)*Temp(i+1,j+1) + KirschNW(3,2)*Temp(i+1,j) + KirschNW(3,1)*Temp(i+1,j-1) + KirschNW(2,1)*Temp(i,j-1);

        % TEST RESULT MATRIX
        % TR = [NW N NE; W 0 E; SW S SE]

        % ARRAY OF RESULT OF KIRSCH MASK
        Kir_Org = [NW N NE E SE S SW W];
        Kir_Sorted = sort(Kir_Org);

        for ki = 1:1:8
            if Kir_Org(1,ki) >= Kir_Sorted(1,4)
                Kir_Org(1,ki) = 1;
            else
                Kir_Org(1,ki) = 0;
            end
        end
        KM(i,j) = 128*Kir_Org(1,1) + 64*Kir_Org(1,2) + 32*Kir_Org(1,3) + 16*Kir_Org(1,4) + 8*Kir_Org(1,5) + 4*Kir_Org(1,6) + 2*Kir_Org(1,7) + Kir_Org(1,8);
    end
end

% Particular Columns/Rows/Cells
KM(1,1) = Temp(1,1);
KM(1,168) = Temp(1,168);
KM(192,1) = Temp(192,1);
KM(192,168) = Temp(192,168);

KM(1,2:167) = Temp(1,2:167);
KM(2:191,168) = Temp(2:191,168);
KM(192,2:167) = Temp(192,2:167);
KM(2:191,1) = Temp(2:191,1);

% Show Kirsh Masked Test Image
figure('name','Kirsh Masked Test Image');
imshow(KM/255);


%Divide into 7*7 blocks, each block 189x168 px
KM = KM(2:190,:);

% Matrix of block images
   blockedM = zeros(27*49,24);
   iM = 1;   

% Making concatenated histogram matrix
for i=1:1:7
   for j=1:1:7
       blockedM((iM-1)*27+1:iM*27,1:24) = KM((i-1)*27+1:i*27,(j-1)*24+1:j*24); 
       iM = iM + 1;
   end
end

for i=1:1:49
    HisTest((i-1)*256+1:i*256,1) = imhist(blockedM((i-1)*27+1:i*27,1:24)/255);    
end

%Show Histogram
figure('name','Concatenated Histograms of Test Image');
plot(HisTest(:,1));

% Projected Matrix Test Histogram
HisTestProj = Wopt'*HisTest;
 
%% Comparision
%Result
comp_Coeff = zeros(1,NoC*NoI);
tmp_res =0;
for i=1:1:NoC*NoI
    for j=1:1:nb
       tmp_res = tmp_res + (HisTestProj(j,1)-MatrixHProj(j,i))*(HisTestProj(j,1)-MatrixHProj(j,i)); 
    end
    comp_Coeff(1,i) = tmp_res;
    tmp_res = 0;
end

[M,I] = min(comp_Coeff);
mess1=sprintf('# %d',I);

Org = reshape(A(:,I),[192,168]);
figure('name','Most Similar Database Image');
imshow(Org);
