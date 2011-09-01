function poeDriver(parameters, folds, jobNum, Outpath)

    scores  = zeros(length(folds), 1);
    native  = zeros(length(folds), 1);
    T       = {};

    for f = 1:length(folds)
        
        if strcmpi(parameters.filter, 'raw')
            C       = folds(f).Ctrain_raw;
            Cval    = folds(f).Ctrain_raw_val;
        elseif strcmpi(parameters.filter, 'noback')
            C       = folds(f).Ctrain_noback;
            Cval    = folds(f).Ctrain_noback_val;
        else
            C       = folds(f).Ctrain_redundant;
            Cval    = folds(f).Ctrain_redundant_val;
        end

        % Sweep over the beta range
        bestScore   = -inf;
        bestBeta    = 0;

        for b = parameters.C
            tr = poe_linear(folds(f).Ktrain(:,:,parameters.kernels), C, 1, ...
                            struct('fullmatrix',    1 - parameters.diagonal, ...
                                    'beta',         b));
            s  = poe_test(tr,   folds(f).Ktrain(:,:,parameters.kernels), ...
                                folds(f).Ktrain(:,:,parameters.kernels), ...
                                Cval);
            if s > bestScore
                bestScore = s;
                bestBeta = b;
            end
        end

        % Got the best beta, retrain with full constraint set
        T{f} = poe_linear(  folds(f).Ktrain(:,:,parameters.kernels), ...
                            [C;Cval], 1, struct(...
                                'fullmatrix',   1 - parameters.diagonal, ...
                                'beta',         bestBeta));
        native(f) = poe_test([], folds(f).Ktrain(:,:,parameters.kernels), ...
                                 folds(f).Ktest(:,:,parameters.kernels), ...
                                 folds(f).Ctest);
        scores(f) = poe_test(T{f}, folds(f).Ktrain(:,:,parameters.kernels), ...
                                 folds(f).Ktest(:,:,parameters.kernels), ...
                                 folds(f).Ctest);
    end

    % Got all the scores, now save output and report

    experimentShowOutput(jobNum, parameters, scores, native);

    save(sprintf('%s/job%02d.mat', Outpath, jobNum), ...
            'parameters', 'scores', 'native', 'T');
end

