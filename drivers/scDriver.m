function scDriver(parameters, data, jobNum, Outpath)

    % parameters:
    %   directory       - where the mfcc files live
    %   file            - where the specific file lives
    %   steps           - number of steps to run coordinate descent

    % data:
    %   mu              - mean & stdev needed for the bzip loader
    %   sigma
    %   C               - codebook
    %   Cdesc           - codebook description string

    % output:
    %   S              - sparse coding output
    %

    % file format:
    %   Outpath/SC/SC-Cdesc-steps-BASENAME.mat
    %       BASENAME = strrep(file, '.mfcc.bz2', '')

    % Load the file
    D = loadBzipDeltaZ( sprintf('%s/%s', parameters.directory, parameters.file), ...
                        data.mu, data.sigma);

    % Compute the sparse coding coefficients
    tic;
        S = SC(D, data.C, parameters.steps);
    S.time         = toc;


    % Check to see if our output directory exists
    if ~exist(sprintf('%s/SC', Outpath), 'dir')
        mkdir(Outpath, 'SC');
    end

    % Save the learned mixture
    filename = strrep(parameters.file, '.mfcc.bz2', '.mat');
    save(sprintf('%s/SC/SC-%s-%d-%s', Outpath, data.Cdesc, ...
                                parameters.steps, ...
                                filename), 'S');
end
