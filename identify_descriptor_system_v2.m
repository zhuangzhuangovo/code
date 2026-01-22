function [E_re, A_re, B_re, C_re, D_re, n_re, x0, j] = ...
    identify_descriptor_system_v2(u, y, i, lambda_reg, opts)
% INPUTS:
%   u          - Input data matrix (m × N)
%   y          - Output data matrix (l × N)
%   i          - Block size of Hankel matrices
%   lambda_reg - Tikhonov regularization parameter
%   opts       - Relative threshold for singular value truncation
%
% OUTPUTS:
%   E_re, A_re, B_re, C_re, D_re - Identified descriptor system matrices
%   n_re       - Identified system order
%   x0         - Estimated initial state
%   j          - Number of usable columns in Hankel matrices

%% ===================== Dimensions ===========================
[m, N] = size(u);
j = N - 2*i + 1;

%% ===================== 1. Hankel matrices ==================
% Combined input-output Hankel matrices
H1_i  = build_hankel_combined(u, y, 1,     i,   j);
Hi_2i = build_hankel_combined(u, y, i+1, 2*i,   j);

M = [H1_i; Hi_2i];

%% ===================== 2. Rank estimation ==================
% SVD-based rank determination
[~, S_M, ~] = svd(M, 'econ');
sigmaM = diag(S_M);

tolM = opts * sigmaM(1);
rM = sum(sigmaM > tolM);

% System order from rank condition: rank(M) = 2mi + n
n = rM - 2*m*i;

[U, ~, ~] = svd(M);

%% ===================== 3. State sequence recovery ==========
% Extract orthogonal complement associated with state subspace
U12 = U(1:size(H1_i,1), rM+1:end);
temp = U12' * H1_i;

[~, St, Vt] = svd(temp, 'econ');

% Estimated lifted state sequence X_{i|i+1}
Xi_i_plus_1 = ...
    diag(sqrt(diag(St(1:n,1:n)))) * Vt(:,1:n)';

%% ===================== 4. Shift invariance ==================
% Least-squares estimation of the shift operator
T = Xi_i_plus_1 * pinv(Hi_2i);

% Shifted Hankel matrix
Hi_2i_minus_1 = build_hankel_combined(u, y, i, 2*i-1, j);

% Shifted state sequence X_{i-1|i}
Xi_minus_1_i = T * Hi_2i_minus_1;

%% ===================== 5. Descriptor system matrices =======
% Null-space condition:
%   [E  A  B] * [X_{i|i+1}; X_{i-1|i}; U_{i|i}] = 0

Ui_i = u(:, i:i+j-1);
K = [Xi_i_plus_1; Xi_minus_1_i; Ui_i];

[UK, SK, ~] = svd(K, 'econ');
sing_vals = diag(SK);

tol_null = 1e-1 * sing_vals(1);
null_dim = sum(sing_vals < tol_null);

if null_dim < n
    warning('Null-space dimension (%d) smaller than system order (%d).', ...
             null_dim, n);
end

% Left null-space basis
sol = UK(:, end-n+1:end)';

E_ptr = sol(:, 1:n);
A_ptr = -sol(:, n+1:2*n);
B_ptr = -sol(:, 2*n+1:2*n+m);

%% ===================== 6. Mode separation ==================
% Generalized eigenvalue analysis
tol_eig = 1e8;
[~, D] = eig(A_ptr, E_ptr);
lambda = diag(D);

finite_idx = isfinite(lambda) & abs(lambda) < tol_eig;
nr = sum(finite_idx);     % Dynamic modes
ns = n - nr;              % Algebraic modes

% Quasi-Weierstrass form
[S, P] = quasi_weierstrass(E_ptr, A_ptr, B_ptr);

E_id = S * E_ptr * P;
A_id = S * A_ptr * P;
B_id = S * B_ptr;

Es = E_id(nr+1:end, nr+1:end);
Ar = A_id(1:nr, 1:nr);
Br = B_id(1:nr, :);
Bs = B_id(nr+1:end, :);

%% ===================== 7. Output equation ==================
% State reconstruction in original coordinates
X1 = inv(P) * Xi_minus_1_i;
X2 = inv(P) * Xi_i_plus_1;

X_ir = X1(1:nr, :);
X_is = X2(nr+1:end, :);

Yi_i = y(:, i:i+j-1);
Phi = [X_ir; X_is; Ui_i];

% Tikhonov-regularized least squares
CD_est = (Yi_i * Phi') / ...
         (Phi * Phi' + lambda_reg * eye(size(Phi,1)));

Cr  = CD_est(:, 1:nr);
Cs  = CD_est(:, nr+1:nr+ns);
Drs = CD_est(:, n+1:end);

%% ===================== 8. Final identified model ===========
E_re = [eye(nr), zeros(nr,ns);
        zeros(ns,nr), Es];

A_re = [Ar, zeros(nr,ns);
        zeros(ns,nr), eye(ns)];

B_re = [Br; -Bs];
C_re = [Cr, Cs];
D_re = Drs;

n_re = nr + ns;

% Estimated initial state
x0 = [X_ir(:,end); X_is(:,end)];

end
