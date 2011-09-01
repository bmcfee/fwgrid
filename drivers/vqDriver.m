function vqDriver(parameters, data, jobNum, Outpath)

    % parameters:
    %   directory       - where the mfcc files live
    %   file            - where the specific file lives
    %   k               - how many codewords per frame

    % data:
    %   mu              - mean & stdev needed for the bzip loader
    %   sigma
    %   C               - codebook
    %   Cdesc           - codebook description string

    % output:
    %   vq              - VQ output
    %

    % file format:
    %   Outpath/VQ/VQ-Ddesc-k-BASENAME.mat
    %       BASENAME = strrep(file, '.mfcc.bz2', '')

    % Load the file
    D = loadBzipDeltaZ( sprintf('%s/%s', parameters.directory, parameters.file), ...
                        data.mu, data.sigma);

    % Compute the mixture
    tic;
        V = VQ(D, data.C, parameters.k);
    V.time        = toc;
    V.filename    = parameters.file;


    % Check to see if our output directory exists
    if ~exist(sprintf('%s/VQ', Outpath), 'dir')
        mkdir(Outpath, 'VQ');
    end

    % Save the learned mixture
    filename = strrep(parameters.file, '.mfcc.bz2', '.mat');
    save(sprintf('%s/VQ/%s-%d-%s', Outpath, data.Cdesc, parameters.k, filename), 'V');
end
