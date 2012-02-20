function itmlDriver(parameters, folds, jobNum, Outpath)

    scores  = zeros(length(folds), 1);
    native  = zeros(length(folds), 1);
    T       = {};

    for f = 1:length(folds)
        
        bestScore   =   -inf;
        bestK       =   0;

        A0          = eye(size(folds(f).Ktrain,1));
        for C = parameters.C
            W = MetricLearning(@ItmlAlg, ...
                                folds(f).Ytrain, folds(f).Ktrain', ...
                                A0, ...
                                struct('gamma', C));

            if ~isempty(folds(f).Yval)
                Perf        = mlr_test( W, parameters.test_k,   ...
                                        folds(f).Ktrain,        ...
                                        folds(f).Ytrain,        ...
                                        folds(f).Kval,          ...
                                        folds(f).Yval);
            else
                Perf        = mlr_test( W, parameters.test_k,   ...
                                        folds(f).Ktrain,        ...
                                        folds(f).Ytrain);
            end
            
%             S   = Perf.KNN;
            S   = Perf.AUC;
            k   = Perf.KNNk;
            if S > bestScore
                bestScore   = S;
                T{f}        = struct('W', W, 'test_k', k);
            end
        end

        perfs{f}    = mlr_test( T{f}.W, T{f}.test_k,    ...
                                [folds(f).Ktrain folds(f).Kval], ...
                                [folds(f).Ytrain ; folds(f).Yval], ...
                                folds(f).Ktest,         ...
                                folds(f).Ytest);
        scores(f)   = perfs{f}.KNN;
            
        % Compute native scores using best validation k
        if ~isempty(folds(f).Yval)
            P           = mlr_test( [], parameters.test_k,   ...
                                    folds(f).Ktrain, folds(f).Ytrain,       ...
                                    folds(f).Kval, folds(f).Yval);
        else
            P           = mlr_test( [], parameters.test_k,   ...
                                    folds(f).Ktrain, folds(f).Ytrain);
        end
        P           = mlr_test( [], P.KNNk,   ...
                                [folds(f).Ktrain folds(f).Kval], ...
                                [folds(f).Ytrain ; folds(f).Yval], ...
                                folds(f).Ktest, folds(f).Ytest);
        native(f)   = P.KNN;
    end

    experimentShowOutput(jobNum, parameters, scores, native);
    save(sprintf('%s/job%02d.mat', Outpath, jobNum), ...
            'parameters', 'scores', 'native', 'T');
end

