function [ evalObj, diff ] = transEval2( M, tstMap, srcVecM, tgtVecM )
%TRANSVAL Summary of this function goes here
%   evaluate M based on predict accauracy and ranking acc@5
%   M learned transform matrix
%   tstMap maping of test set

projVecM = srcVecM*M';

projTstVecM = projVecM(tstMap(tstMap(:,3) == 1,1),:);
tgtTstVecM = tgtVecM(tstMap(tstMap(:,3) == 1,2),:);

diff = norm(projTstVecM - tgtTstVecM, 'fro')^2;

tgtVecM = normr(tgtVecM);
simM = projVecM*tgtVecM';

map = zeros(10, 1);

tgtIdxs = unique(tstMap(:,2));
N = size(tgtIdxs, 1);

for i = 1:N
    tgtIdx = tgtIdxs(i);
%     disp(['tgtIdx: ', sprintf('%d ', tgtIdx)]);
    candSrcIdx = tstMap(tstMap(:,2) == tgtIdx,1);
%     disp(['candSrcIdx: ', sprintf('%d ', candSrcIdx)]);
    goldSrcIdx = tstMap((tstMap(:,2) == tgtIdx) & (tstMap(:,3) == 1),1);
%     disp(['goldSrcIdx: ', sprintf('%d ', goldSrcIdx)]);
    sim = simM(:,tgtIdx);
    [~, rank] = sort(sim,'descend');
    predSrcIdx = rank(ismember(rank, candSrcIdx));
%     disp(['predSrcIdx: ', sprintf('%d ', predSrcIdx)]);
    for j = 1:size(map,1)
        map(j) = map(j) + averagePrecisionAtK(goldSrcIdx, predSrcIdx, j);
%         disp(['ap', num2str(j), ' :', num2str(averagePrecisionAtK(goldSrcIdx, predSrcIdx, j))])
    end
end

evalObj.map = map./N;
evalObj.simM = simM;

end

function score = averagePrecisionAtK(actual, prediction, k)
%AVERAGEPRECISIONATK   Calculates the average precision at k
%   score = averagePrecisionAtK(actual, prediction, k)
%
%   actual is a vector
%   prediction is a vector
%   k is an integer
%
%   Author: Ben Hamner (ben@benhamner.com)

if nargin<3
    k=10;
end

if length(prediction)>k
    prediction = prediction(1:k);
end

score = 0;
num_hits = 0;
for i=1:min(length(prediction), k)
    if sum(actual==prediction(i))>0 && ...
            sum(prediction(1:i-1)==prediction(i))==0
        num_hits = num_hits + 1;
        score = score + num_hits / i;
    end
end

score = score / min(length(actual), k);

end

