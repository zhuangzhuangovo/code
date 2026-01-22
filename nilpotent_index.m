function k = nilpotent_index(N)
% NILPOTENT_INDEX Computes the index of a nilpotent matrix
%
% Input:
%   N : n×n square nilpotent matrix
%
% Output:
%   k : smallest positive integer such that N^k ≈ 0 (within tolerance)

tol = 1e-10;  % Numerical tolerance
n = size(N, 1);

% Ensure input is square
if n ~= size(N, 2)
    error('Input must be a square matrix.');
end

M = eye(n);  % Initialize accumulator as identity matrix
for k = 1:n
    M = M * N;  % Compute N^k
    if max(abs(M(:))) < tol  % Check if all elements are close to zero
        fprintf('Nilpotent index is: %d\n', k);
        return;
    end
end

error('Matrix did not become zero within %d powers. It may not be nilpotent.', n);
end
