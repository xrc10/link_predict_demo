clc; clear all; close all;

%% read src vector
srcVec = dlmread('data/en.svm');
srcVec = srcVec(:,(2:end));

%% read tgt vector
% tgtVec = dlmread('data/fr.norm.svm');
tgtVec = dlmread('data/ha.norm.svm');
tgtVec = tgtVec(:,(2:end));

%% read trnMap and tstMap
trnMap = dlmread('data/dict.ha.trn.txt');
trnMap = trnMap(trnMap(:,3)==1,:);

valMap = dlmread('data/dict.ha.val.txt');

%% train M
regType = 2;
lambda = 1;
M = transLearnMatInv(srcVec(trnMap(:,2),:), tgtVec(trnMap(:,1),:), lambda, regType);

%% evaluation
[evalObj, diff] = transEval2(M, [valMap(:,2), valMap(:,1), valMap(:,3)], srcVec, tgtVec);

%% print results
evalString = sprintf('%f ', evalObj.map');
fprintf('map@1-10:%s\n', evalString);

%% save simMat
fprintf('saving simM...\n');
simM = evalObj.simM;
sps = 10; % make it sparse
for k = 1:size(simM, 2)
    [~, idx] = sort(simM(:,k), 'descend');
    simM(idx(sps+1:end),k) = 0;
end
simM = sparse(simM);
save('data/simM', 'simM', '-v7.3');




