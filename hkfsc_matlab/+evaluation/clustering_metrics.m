function metrics = clustering_metrics(truth, prediction)
%CLUSTERING_METRICS External ACC, NMI, and ARI evaluation.
truth = relabel(truth(:)); prediction = relabel(prediction(:));
if numel(truth) ~= numel(prediction), error('HKFSC:MetricSize','Label lengths differ.'); end
kt=max(truth); kp=max(prediction); k=max(kt,kp);
C=zeros(k);
for i=1:numel(truth), C(truth(i),prediction(i))=C(truth(i),prediction(i))+1; end
assignment=hungarian_max(C);
matched=0;
for i=1:k, if assignment(i)>0, matched=matched+C(i,assignment(i)); end, end
metrics.ACC=matched/numel(truth);
mapped=zeros(size(prediction));
for i=1:k
    if assignment(i)>0,mapped(prediction==assignment(i))=i;end
end
unmapped=mapped==0;mapped(unmapped)=prediction(unmapped);
metrics.matched_prediction=mapped;
metrics.truth_relabelled=truth;

P=C/sum(C(:)); pt=sum(P,2); pp=sum(P,1);
den=pt*pp; take=P>0 & den>0;
mi=sum(P(take).*log(P(take)./den(take)));
ht=-sum(pt(pt>0).*log(pt(pt>0))); hp=-sum(pp(pp>0).*log(pp(pp>0)));
metrics.NMI=mi/max(sqrt(ht*hp),eps);

n=sum(C(:)); sum_pairs=sum(sum(C.*(C-1)/2));
row_pairs=sum(sum(C,2).*(sum(C,2)-1)/2);
col_pairs=sum(sum(C,1).*(sum(C,1)-1)/2);
total_pairs=n*(n-1)/2;
expected=row_pairs*col_pairs/max(total_pairs,eps);
upper=0.5*(row_pairs+col_pairs);
metrics.ARI=(sum_pairs-expected)/max(upper-expected,eps);
end

function y=relabel(y)
[~,~,y]=unique(y,'sorted');
end

function assignment=hungarian_max(score)
cost=max(score(:))-score;
n=size(cost,1);
cost=cost-min(cost,[],2); cost=cost-min(cost,[],1);
star=false(n); prime=false(n); rowcov=false(n,1); colcov=false(n,1);
for i=1:n
    for j=1:n
        if abs(cost(i,j))<1e-12 && ~rowcov(i) && ~colcov(j)
            star(i,j)=true; rowcov(i)=true; colcov(j)=true;
        end
    end
end
rowcov(:)=false; colcov=any(star,1)';
while sum(colcov)<n
    [r,c]=find_uncovered_zero(cost,rowcov,colcov);
    while r==0
        block=cost(~rowcov,~colcov); delta=min(block(:));
        if isempty(delta) || ~isfinite(delta), break; end
        cost(rowcov,:)=cost(rowcov,:)+delta;
        cost(:,~colcov)=cost(:,~colcov)-delta;
        [r,c]=find_uncovered_zero(cost,rowcov,colcov);
    end
    if r==0, break; end
    prime(r,c)=true;
    sc=find(star(r,:),1);
    if isempty(sc)
        path=[r c]; done=false;
        while ~done
            rr=find(star(:,path(end,2)),1);
            if isempty(rr), done=true; else
                path=[path;rr path(end,2)]; %#ok<AGROW>
                cc=find(prime(rr,:),1);
                path=[path;rr cc]; %#ok<AGROW>
            end
        end
        for q=1:size(path,1), star(path(q,1),path(q,2))=~star(path(q,1),path(q,2)); end
        prime(:)=false; rowcov(:)=false; colcov=any(star,1)';
    else
        rowcov(r)=true; colcov(sc)=false;
    end
end
assignment=zeros(n,1);
for i=1:n
    j=find(star(i,:),1); if ~isempty(j), assignment(i)=j; end
end
end

function [r,c]=find_uncovered_zero(cost,rowcov,colcov)
mask=abs(cost)<1e-12; mask(rowcov,:)=false; mask(:,colcov)=false;
[r,c]=find(mask,1); if isempty(r), r=0;c=0; end
end
