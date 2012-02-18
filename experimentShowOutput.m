function experimentShowOutput(jobNum, parameters, scores, varargin)

    display('--------------------------------------------------------');
    display(sprintf('Job %02d', jobNum));
    display(parameters);
    display(sprintf('Results:'));
    display(sprintf('\tOptimized: %0.3f (%0.3f)', mean(scores), std(scores)));
    if nargin > 3
        display(sprintf('\tNative: %0.3f (%0.3f)', mean(varargin{1}), std(varargin{1})));
    end
end
