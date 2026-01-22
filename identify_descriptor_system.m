function [E_re, A_re, B_re, C_re, D_re, n_re, x0, j] = identify_descriptor_system(u, y, i)
% IDENTIFY_DESCRIPTOR_SYSTEM Identify a discrete-time descriptor system from input-output data
%
% Inputs:
%   u : m x N input data
%   y : l x N output data
%   i : block row size for Hankel matrices
%
% Outputs:
%   E_re, A_re, B_re, C_re, D_re : identified descriptor system matrices
%   n_re : system order
%   x0   : initial state estimate
%   j    : column size of Hankel matrices

[m, N] = size(u); 
[l, ~] = size(y);
j = N - 2*i + 1;  % columns for Hankel matrices

% --- 1. Construct block Hankel matrices ---
H1_i   = build_hankel_combined(u, y, 1, i, j);      % H_{1|i}
Hi_2i  = build_hankel_combined(u, y, i+1, 2*i, j);  % H_{i+1|2i}

% --- 2. Compute intersection of row spaces (X'_{i|i+1}) ---
M = [H1_i; Hi_2i];
[U, ~, ~] = svd(M);
n = rank(M) - 2*m*i;  % system order

U12 = U(1:size(H1_i,1), (2*m*i + n)+1:end);
temp = U12' * H1_i;
[~, St, Vt] = svd(temp, 'econ');
Xi_i_plus_1 = diag(sqrt(diag(St(1:n,1:n)))) * Vt(:,1:n)'; % weighted singular vectors

% --- 3. Compute X'_{i-1|i} via transformation T ---
T = Xi_i_plus_1 * pinv(Hi_2i);  
Hi_2i_minus_1 = build_hankel_combined(u, y, i, 2*i-1, j);
Xi_minus_1_i = T * Hi_2i_minus_1;

% --- 4. Solve for system matrices E', A', B' ---
Ui_i = u(:, i:i+j-1); % input block
K = [Xi_i_plus_1; Xi_minus_1_i; Ui_i];
[UK, SK, ~] = svd(K, 'econ');

% Zero-space tolerance and dimension
tol_nullspace = 1e-6 * SK(1,1);
zero_space_dim = sum(diag(SK) < tol_nullspace);
if zero_space_dim < n
    warning('Zero-space dimension (%d) less than system order (%d)', zero_space_dim, n);
end

sol = UK(:, end-n+1:end)'; % left null-space basis
E_ptr = sol(:, 1:n);
A_ptr = -sol(:, n+1:2*n);
B_ptr = -sol(:, 2*n+1:2*n+m);

% --- 5. Quasi-Weierstrass transformation for descriptor form ---
tol_eigen = 1e8;
[V, D] = eig(A_ptr, E_ptr);
finite_idx = isfinite(diag(D)) & abs(diag(D)) < tol_eigen;
nr = sum(finite_idx);          % regular part
ns = n - nr;      % singular part

[S, P] = quasi_weierstrass(E_ptr, A_ptr, B_ptr);
E_id = S * E_ptr * P;
A_id = S * A_ptr * P;
B_id = S * B_ptr;

Es = E_id(nr+1:end, nr+1:end);
Ar = A_id(1:nr, 1:nr);
Br = B_id(1:nr, :);
Bs = B_id(nr+1:end, :);

Yi_i = y(:, i:i+j-1);
X_ir = inv(P) * Xi_minus_1_i; X_ir = X_ir(1:nr,:);
X_is = inv(P) * Xi_i_plus_1;  X_is = X_is(end-ns+1:end,:);

CD_est = Yi_i * pinv([X_ir; X_is; Ui_i]);
Cr = CD_est(:, 1:nr);
Cs = CD_est(:, nr+1:nr+ns);
Drs = CD_est(:, n+1:end);

% --- 6. Assemble identified system ---
E_re = [eye(nr), zeros(nr,ns); zeros(ns,nr), Es];
A_re = [Ar, zeros(nr,ns); zeros(ns,nr), eye(ns)];
B_re = [Br; -Bs];
C_re = [Cr, Cs];
D_re = Drs;
n_re = nr + ns;

% Initial state
x0 = [X_ir(:,end); X_is(:,end)];
end
