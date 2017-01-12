%% load AMP data
% Original dataset
load AMPdata.mat  % samples, labels
labels(labels==0) = -1;

% % Filtered dataset
% load AMPdata_filtered.mat  % APdata, Type:struct
% numSample = length(APdata);
% samples = [];
% labels = [];
% for i = 1:numSample
%     samples = [ samples; [APdata(i).AAC, APdata(i).DC] ];
%     labels = [ labels; APdata(i).LabelVec ];
% end
% clear APdata
% labels(labels==0) = -1;

%% Begin cross-validation
k = 15;   % number of nearest neighbors
lambda = 1;  % regularization parameter
n = 5;    % n-fold
times = 1;   % Times of CV
dispflag = true;

disp(['Times of CV: ' num2str(times)])
disp(['Folds of CV: ' num2str(n)])
for i = 1:times
    disp(['============== Times ' num2str(i) '/' num2str(times)])
    [HammingLoss(i), SubsetAccuracy(i), AveragePrecision(i), Coverage(i), OneError(i), RankingLoss(i), microF1(i), macroF1(i)] ...
        = MLSWKnnCV(samples, labels, k, lambda, n, dispflag);
end

%% print result
meanResult = struct;
meanResult.HammingLoss = [ mean(HammingLoss)  std(HammingLoss) ];
meanResult.SubsetAccuracy = [ mean(SubsetAccuracy)  std(SubsetAccuracy) ];
meanResult.AveragePrecision = [ mean(AveragePrecision)  std(AveragePrecision) ];
meanResult.Coverage = [ mean(Coverage)  std(Coverage) ];
meanResult.OneError = [ mean(OneError)  std(OneError) ];
meanResult.RankingLoss = [ mean(RankingLoss)  std(RankingLoss) ];
meanResult.microF1 = [ mean(microF1)  std(microF1) ];
meanResult.macroF1 = [ mean(macroF1)  std(macroF1) ];
disp(meanResult)
