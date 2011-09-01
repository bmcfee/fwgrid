function smdDriver(parameters, data, jobNum, Outpath)

    % parameters:
    %   directory       - where the mfcc files live
    %   file            - where the specific file lives

    % data:
    %   vocabMix

    % output:
    %   smd             - posteriors of tags given track

    % file format:
    %   Outpath/SMD/SMD-BASENAME.mat
    %       BASENAME = strrep(file, '.mfcc.bz2', '')

    % Load the file
    M = loadBzipMFCC( sprintf('%s/%s', parameters.directory, parameters.file));

    % Compute the delta, delta^2
    D   = deltaWindow(M');
    DD  = deltaWindow(D);
    M   = [M D' DD'];
    
    % Compute log-likelihood
    LL  = wordModelInference(data.vocabMix, M);

    % Convert to posteriors

    smd = exp(LL);
    smd = smd / sum(smd);

    % Check to see if our output directory exists
    if ~exist(sprintf('%s/SMD', Outpath), 'dir')
        mkdir(Outpath, 'SMD');
    end

    % Save the learned mixture
    filename = strrep(parameters.file, '.mfcc.bz2', '.mat');
    save(sprintf('%s/SMD/SMD-%s', Outpath, filename), 'smd');
end
