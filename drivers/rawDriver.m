function rawDriver(parameters, folds, jobNum, Outpath)

    scores  = zeros(length(folds),1);

    for f = 1:length(folds)

        if ~isempty(folds(f).Kval)
            P           = mlr_test( [], parameters.test_k,   ...
                                    folds(f).Ktrain(:,:,parameters.kernels), folds(f).Ytrain,       ...
                                    folds(f).Kval(:,:,parameters.kernels), folds(f).Yval);
        else
            P           = mlr_test( [], parameters.test_k,   ...
                                    folds(f).Ktrain(:,:,parameters.kernels), folds(f).Ytrain);
        end

        P           = mlr_test( [], P.KNNk,         ...
                                [folds(f).Ktrain(:,:,parameters.kernels) folds(f).Kval(:,:,parameters.kernels)],    ...
                                [folds(f).Ytrain ; folds(f).Yval],    ...
                                folds(f).Ktest(:,:,parameters.kernels), folds(f).Ytest);
        scores(f)   = P.KNN;
    end

    experimentShowOutput(jobNum, parameters, scores);
    save(sprintf('%s/job%02d.mat', Outpath, jobNum), ...
            'parameters', 'scores');
end

