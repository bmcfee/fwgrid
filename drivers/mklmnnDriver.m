function mklmnnDriver(parameters, folds, jobNum, Outpath)

    scores  = zeros(length(folds), 1);
    native  = zeros(length(folds), 1);
    T       = {};

    for f = 1:length(folds)
        
        % Validate over training neighborhood size and regularization

        bestScore   = -inf;
        bestTrainK  = 0;
        bestC       = 0;
        bestTestK   = 0;

        for k = parameters.train_k
            for C = parameters.C
                tr      = mklmnn_train( folds(f).Ktrain(:,:,parameters.kernels), ...
                                        folds(f).Ytrain, ...
                                        struct('diagonal', parameters.diagonal, ...
                                                'reg', parameters.reg, ...
                                                'k', k, ...
                                                'c', C));

                if ~isempty(folds(f).Yval)
                    Perf        = mlr_test(tr.C,    parameters.test_k,  ...
                                                    folds(f).Ktrain(:,:,parameters.kernels),    ...
                                                    folds(f).Ytrain,    ...
                                                    folds(f).Kval(:,:,parameters.kernels),      ...
                                                    folds(f).Yval);
                else
                    Perf        = mlr_test(tr.C,    parameters.test_k,  ...
                                                    folds(f).Ktrain(:,:,parameters.kernels),    ...
                                                    folds(f).Ytrain);
                end

                if Perf.KNN > bestScore
                    bestScore   = Perf.KNN;
                    bestTrainK  = k;
                    bestTestK   = Perf.KNNk;
                    bestC       = C;
                    T{f}        = struct('W', tr.C, 'tr', tr, 'test_k', bestTestK, 'C', bestC);
                end
            end
        end

        perfs{f}    = mlr_test( T{f}.W, T{f}.test_k,    ...
                                [folds(f).Ktrain(:,:,parameters.kernels) folds(f).Kval(:,:,parameters.kernels)], ...
                                [folds(f).Ytrain ; folds(f).Yval], ...
                                folds(f).Ktest(:,:,parameters.kernels),         ...
                                folds(f).Ytest);
        scores(f)   = perfs{f}.KNN;

        % Compute native scores using best validation k
        if ~isempty(folds(f).Yval)
            P           = mlr_test( [], parameters.test_k,   ...
                                    folds(f).Ktrain(:,:,parameters.kernels), folds(f).Ytrain,       ...
                                    folds(f).Kval(:,:,parameters.kernels), folds(f).Yval);
        else
            P           = mlr_test( [], parameters.test_k,   ...
                                    folds(f).Ktrain(:,:,parameters.kernels), folds(f).Ytrain);
        end

        P           = mlr_test( [], P.KNNk,         ...
                                [folds(f).Ktrain(:,:,parameters.kernels) folds(f).Kval(:,:,parameters.kernels)],    ...
                                [folds(f).Ytrain; folds(f).Yval],    ...
                                folds(f).Ktest(:,:,parameters.kernels), folds(f).Ytest);
        native(f)   = P.KNN;
    end

    experimentShowOutput(jobNum, parameters, scores, native);
    save(sprintf('%s/job%02d.mat', Outpath, jobNum), ...
            'parameters', 'scores', 'native', 'T');
end

