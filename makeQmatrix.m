function Q = makeQmatrix(pol_indx, PI)
% computes matrix Q for policy function iteration
%
% pol_indx: m-by-n matrix of indices of the endogenous variable chosen
% according to the policy function
%
% PI: m-by-m stochastic transition matrix of the exogenous state variable
%
% Q: the m*n-by-m*n matrix to compute expected value of following the
% policy

m = length(PI);
n = size(pol_indx, 2);
Q = sparse(m*n, m*n);

% sequential way to construct Q:
for xi = 1:n
    for zi = 1:m
        row = (xi - 1) * m +  zi;
        col = (pol_indx(zi, xi) - 1) * m;
        Q(row, col+1:col+m) = PI(zi,:);
    end
end

% alternative, vectorized way:
rows = kron((1:m*n)', ones(m,1));
cols = kron((pol_indx(:) - 1) * m, ones(m,1)) + kron(ones(m*n, 1), (1:m)');
% vals = repmat([PI(1,:)'; PI(2,:)'], [n, 1]);
vals = repmat( reshape( permute( PI, [2 1] ), [], 1) , [n, 1]);
Q = sparse(rows, cols, vals, m*n, m*n);



