function [predLabel, output] = MLSWKnnPred(trainSample, trainLabel, testSample, K, simMea, W)
% Multi-label predicting based on Distance-weighted k-nearest Neighbor and linear regression
%
% trainSample - training samples
% trainLabel - training samples' label vectors
% testSample - testing samples
% K - the number of nearest neighbors
% simMea - similarity measure method
% W - the matrix of regression coefficients obtained from MLSWKnnTrain()
% predLabel - Predicted labels for testing samples
% output - outputs of testing samples for each label

[~, numClass] = size(trainLabel);
numTest = size(testSample,1);

%% Begin Knn
Z = zeros(numTest, numClass);
% for each test sample, do:
for i = 1:numTest
    % get simlarities with all training samples
    sims = pdist2(testSample(i,:), trainSample, simMea);
    [simSorted, ind] = sort(sims,'descend');
    labelKnn = trainLabel(ind(1:K),:);
    simKnn = simSorted(1:K);
    maxSim = max(simKnn);  minSim = min(simKnn);
    if maxSim ~= minSim
        simKnn = (simKnn-minSim) / (maxSim-minSim);
    end
    %calculate score vector
     Z(i,:) = simKnn*(labelKnn==1)/sum(simKnn);
end

%Predicted Labels and outputs
Z = [ones(numTest,1) Z];
output = Z*W;
predLabel = 2*(output>=0)-1;
end
