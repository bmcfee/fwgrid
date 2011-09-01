function gmmDriver(parameters, data, jobNum, Outpath)

    % parameters:
    %   directory       - where the mfcc files live
    %   file            - where the specific file lives
    %   nmix            - how many mixture components

    % data:
    %   mu              - mean & stdev needed for the bzip loader
    %   sigma

    % output:
    %   mix             - gmm output
    %   logp            - gmm output
    %   training time   - time to train this gmm
    %

    % file format:
    %   Outpath/GMM/GMM-NMIX-BASENAME.mat
    %       BASENAME = strrep(file, '.mfcc.bz2', '')

    % Load the file
    D = loadBzipDeltaZ( sprintf('%s/%s', parameters.directory, parameters.file), ...
                        data.mu, data.sigma);

    % Compute the mixture
    tic;
        [mix, logp] = GMM_EM2(D, parameters.nmix);
    mix.time        = toc;
    mix.logp        = logp;
    mix.filename    = parameters.file;


    % Check to see if our output directory exists
    if ~exist(sprintf('%s/GMM', Outpath), 'dir')
        mkdir(Outpath, 'GMM');
    end

    % Save the learned mixture
    filename = strrep(parameters.file, '.mfcc.bz2', '.mat');
    save(sprintf('%s/GMM/%s', Outpath, filename), 'mix');
end
