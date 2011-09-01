function gmmRankingDriver(parameters, data, jobNum, Outpath)

    % parameters:
    %   fold        - which fold number
    %   samples     - how many samples in cross-entropy approximation

    perfs = testGMMretrieval(   data{parameters.fold}.train_data,   ...
                                data{parameters.fold}.test_data,    ...
                                data{parameters.fold}.test_Y,       ...
                                parameters.samples);

    save(sprintf('%s/job%02d.mat', Outpath, jobNum), 'parameters', 'perfs');

end
