function parameters = mkparams(varargin)

    if mod(nargin, 2) ~= 0
        error('Incorrect argument count');
    end

    Q = reshape(varargin, 2, nargin / 2)';

    parameters = recurseParams(Q);

    parameters = parameters(:);
end

function P = recurseParams(Q)

    % Base case, only one param
    if size(Q,1) == 1
        P = struct(Q{1,1}, Q{1,2}(:));
        return;
    end

    % Otherwise, recurse
    P2 = recurseParams(Q(2:end,:));

    % Now, make copies of P2 and fill in with each value of Q{1,2}

    n = length(Q{1,2});
    m = length(P2);
    P2 = repmat(P2, n, 1);

    z = 1;
    for i = 1:n
        for j = 1:m
            P(z) = setfield(P2(z), Q{1,1}, Q{1,2}{i});
            z = z + 1;
        end
    end
end
