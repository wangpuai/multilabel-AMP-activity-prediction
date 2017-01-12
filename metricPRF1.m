function [Precision, Recall, F1] = metricPRF1(X, Y, type)
% X - numClass��numSample predicted label, 0-1 matrix
% Y - numClass��numSample target label, 0-1 matrix
% type - 'micro' or 'macro'

XandY = X & Y;

if strcmp(type,'micro')
    if sum(XandY(:))==0
        Precision = 0;
        Recall = 0;
        F1 = 0;
    elseif sum(X(:))==0
        Precision = 0;
        Recall = 0;
        F1 = 0;
    elseif sum(Y(:))==0
        Precision = 0;
        Recall = 0;
        F1 = 0;
    else
        Precision = sum(XandY(:)) / sum(X(:));
        Recall = sum(XandY(:)) / sum(Y(:));
        F1 = 2*Precision*Recall / (Precision+Recall);
    end
elseif strcmp(type,'macro')
    p = sum(XandY,2) ./ sum(X,2);  % precision for each label, numClass��1
    r = sum(XandY,2) ./ sum(Y,2);  % recall for each label, numClass��1
    p(isnan(p)) = 0;
    r(isnan(r)) = 0;

    f = 2*p.*r./(p+r);    % f1 for each label, numClass��1
    f(isnan(f)) = 0;

    Precision = mean(p);
    Recall = mean(r);
    F1 = mean(f);
end

%--------------------------------------------------------------------------
% TPi = sum( X(i,:) & Y(i,:) )
% TPi + FPi = sum( X(i,:) )
% TPi + FNi = sum( Y(i,:) )
% for macro:
% PRECISIONi = TPi / (TPi + FPi)
% RECALLi = TPi / (TPi + FNi)
% F1i = 2TPi / (2TPi + FPi + FNi)
%     = 2PRECISIONi*RECALLi/(PRECISIONi+RECALLi)
% for micro:
% TP = ��TPi,  FP = ��FPi,  FN = ��FNi
% PRECISION = TP / (TP + FP) = sum(XandY) / sum(X(:))
% RECALL = TP / (TP + FN) = sum(XandY) / sum(Y(:))
% F1 = 2TP / (2TP + FP + FN)
%    = 2*PRECISION*RECALL/(PRECISION+RECALL)
%--------------------------------------------------------------------------
