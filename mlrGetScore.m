function [S,k] = mlrGetScore(Perf, parameters)

    k = 0;
    if strcmpi(parameters.score, 'auc')
        S = Perf.AUC;
    elseif strcmpi(parameters.score, 'knn')
        S = Perf.KNN;
        k = Perf.KNNk;
    elseif strcmpi(parameters.score, 'precatk')
        S = Perf.PrecAtK;
        k = Perf.PrecAtKk;
    elseif strcmpi(parameters.score, 'map')
        S = Perf.MAP;
    elseif strcmpi(parameters.score, 'mrr')
        S = Perf.MRR;
    elseif strcmpi(parameters.score, 'ndcg')
        S = Perf.NDCG;
        k = Perf.NDCGk;
    else
        error(['Unknown score function: ', parameters.score]);
    end
end
