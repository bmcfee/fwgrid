function performExperiment(parameters, folds, jobNum, Outpath)

    fName = [lower(parameters.algorithm), 'Driver'];

    if exist(fName) == 2
        DRIVER = str2func(fName);
        DRIVER(parameters, folds, jobNum, Outpath);
    else
        error(sprintf('Unknown driver: %s', fName));
    end

end
