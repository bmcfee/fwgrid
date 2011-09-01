function mlrDriver(parameters, folds, jobNum, Outpath)

    scores  = zeros(length(folds),1);
    perfs   = {};
    T       = {};

    for f = 1:length(folds)
        if ~isfield(folds(f), 'Kvalnorm')
            folds(f).Kvalnorm = [];
            folds(f).Ktestnorm = [];
        end

        % Validate over C
        bestScore   = -inf;
        bestC       = 0;
        bestTrainK  = 0;
        bestTestK   = 0;
        bestLoss    = 0;

        if ~iscell(parameters.loss)
            parameters.loss = {parameters.loss};
        end
        for loss = parameters.loss
            if iscell(loss)
                loss = cell2mat(loss);
            end
            if  strcmpi(loss, 'auc') || ...
                strcmpi(loss, 'map') || ...
                strcmpi(loss, 'mrr')
                train_k_values = parameters.train_k(end);
            else
                train_k_values = parameters.train_k;
            end

            for C = parameters.C
                for train_k = train_k_values

                    [W, Xi, D]  = mlr_train(folds(f).Ktrain(:,:,parameters.kernels), ...
                                            folds(f).Ytrain, ...
                                            C, ...
                                            loss, ...
                                            train_k, ...
                                            parameters.reg, ...
                                            parameters.diagonal);

                    Perf        = mlr_test( W, parameters.test_k, ...
                                            folds(f).Ktrain(:,:,parameters.kernels), ...
                                            folds(f).Ytrain, ...
                                            folds(f).Kval(:,:,parameters.kernels), ...
                                            folds(f).Yval);

                    [S, k]      = mlrGetScore(Perf, parameters);
                    
                    if S > bestScore
                        bestScore = S;
                        bestC = C;
                        bestTrainK = train_k;
                        bestTestK = k;
                        bestLoss = loss;
                        T{f} = struct('W', W, 'D', D, 'train_k', bestTrainK, 'test_k', bestTestK, 'loss', bestLoss);
                    end
                end
            end
        end

        perfs{f}    = mlr_test( T{f}.W, bestTestK,                               ...
                                [folds(f).Ktrain(:,:,parameters.kernels) folds(f).Kval(:,:,parameters.kernels)],    ...
                                [folds(f).Ytrain ; folds(f).Yval],                            ...
                                folds(f).Ktest(:,:,parameters.kernels),     ...
                                folds(f).Ytest);
        scores(f)   = mlrGetScore(perfs{f}, parameters);


    end

    experimentShowOutput(jobNum, parameters, scores);

    save(sprintf('%s/job%02d.mat', Outpath, jobNum), ...
            'parameters', 'scores', 'perfs', 'T');

end


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
