function W = MLSWKnnTrain(trainSample, trainLabel, K, simMea, lambda)
% Multi-label training based on Weighted k-nearest Neighbor and linear regression
%
% trainSample - training samples
% trainLabel  - training samples' label vectors
% K - the number of nearest neighbors
% simMea - similarity measure method
% lam - regularization parameter
% W - the matrix of regression coefficients

[numTrain, numClass] = size(trainLabel);
% Computing similarities between training samples
simMat = squareform( pdist(trainSample,simMea) );  % numTrain¡ÁnumTrain
simMat(1:numTrain+1:end) = 0;

%% Begin Knn
Z = zeros(numTrain, numClass);
%for each training sample, do:
for i = 1:numTrain
    %sort the similarity
    [simSorted, ind] = sort(simMat(i,:),'descend');
    labelKnn = trainLabel(ind(1:K),:);
    simKnn = simSorted(1:K);
    maxSim = max(simKnn);  minSim = min(simKnn);
    if maxSim ~= minSim
        simKnn = (simKnn-minSim) / (maxSim-minSim);
    end
    %calculate score vector
    Z(i,:) = simKnn*(labelKnn==1)/sum(simKnn);
end

%% get W
Z = [ones(numTrain,1) Z];
W = (Z'*Z + lambda*eye(numClass+1))\(Z'*trainLabel);

