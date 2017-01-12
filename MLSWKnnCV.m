function [HammingLoss, SubsetAccuracy, AveragePrecision, Coverage, OneError, RankingLoss, microF1, macroF1] ...
    = MLSWKnnCV(samples, labels, k, lambda, n, dispflag)
% n-fold Cross-Validation with MLSWKnn algorithm
% 
% samples - A matrix of shape (N, D) containing all samples; there are N samples each of dimension D
% labels  - A matrix of shape (N, Q) containing label vectors for all samples
% k - number of nearest neighbors
% lambda - regularization parameter
% n - n-fold
% dispflag - flag for displaying the folds

%% Begin cross-validation
simMea = @simFun;

if nargin < 6, dispflag=0; end
[N, Q] = size(labels);
PredLabels = zeros(N, Q);
Outputs    = zeros(N, Q);
CVO = cvpartition(N,'kfold',n);
for i = 1:CVO.NumTestSets
    if dispflag
        disp(['Fold ' num2str(i) '/' num2str(n)])
    end
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    W = MLSWKnnTrain(samples(trIdx,:), labels(trIdx,:), k, simMea, lambda);
    [~, iOutputs] = MLSWKnnPred(samples(trIdx,:), labels(trIdx,:), samples(teIdx,:), k, simMea, W);
    iPredLabels = 2*(iOutputs>=0)-1;
    PredLabels(teIdx,:) = iPredLabels;
    Outputs(teIdx,:)    = iOutputs;
end

%% Evaluation
%Example-based ranking metrics
RankingLoss      = Metric_RankingLoss(Outputs', labels');
OneError         = Metric_OneError(Outputs', labels');
Coverage         = Metric_Coverage(Outputs', labels');
AveragePrecision = Metric_AveragePrecision(Outputs', labels');

%Example-based classification metrics
labels = (labels+1)/2;
PredLabels = (PredLabels+1)/2;
HammingLoss    = sum(sum(xor(labels,PredLabels)))/(N*Q);
SubsetAccuracy = sum(all(labels==PredLabels,2))/N;
[~, ~, microF1] = metricPRF1(PredLabels, labels, 'micro');
[~, ~, macroF1] = metricPRF1(PredLabels, labels, 'macro');
