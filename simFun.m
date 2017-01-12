function sims = simFun(XI, XJ)
% similarity function
% taking as arguments a 1-by-N vector XI containing a single row of X, an
% M2-by-N matrix XJ containing multiple rows of X, and returning an
% M2-by-1 vector of similarities sims, whose Jth element is the similarity
% between the observations XI and XJ(J,:).


% sim(Vec1,Vec2) = sum(min(Vec1,Vec2))/sum(max(Vec1,Vec2))
num = sum( min(repmat(XI,size(XJ,1),1), XJ), 2 );   % M2-by-1
den = sum( max(repmat(XI,size(XJ,1),1), XJ), 2 );   % M2-by-1

sims = num./den;   % M2-by-1
