function [S_out, P_out, n1, n2] = quasi_weierstrass(E, A, B)
% QUASI_WEIERSTRASS Transforms a descriptor system (E,A) into quasi-Weierstrass form
%
% Inputs:
%   E : n×n descriptor matrix
%   A : n×n state matrix
%   B : n×m input matrix (optional, not used in computation here)
%
% Outputs:
%   S_out : Transformation matrix S
%   P_out : Transformation matrix P
%   n1    : Dimension of the regular (differential) part
%   n2    : Dimension of the singular (algebraic) part

n = size(E, 1);

% --- Step 1: Convert to generalized Schur form ---
sys = dss(A, B, [], [], E);          % Create descriptor state-space system
[syssch, Q, Z, ev] = slgsrsf(sys);  % Compute generalized Schur form

E_1 = syssch.E;
A_1 = syssch.A;

% --- Step 2: Select differential part based on diagonal entries ---
tol = max(size(E_1)) * eps(norm(E_1,'fro'));
sel = abs(diag(E_1)) > tol;

% --- Step 3: Reorder generalized Schur form ---
[A_ord, E_ord, Q_ord, Z_ord] = ordqz(A_1, E_1, Q', Z, sel);

% --- Step 4: Determine block dimensions ---
d  = sum(sel);
n1 = d;          % Differential (regular) part
n2 = n - d;      % Algebraic (singular) part

% --- Step 5: Extract blocks ---
E11 = E_ord(1:n1,    1:n1);  
E12 = E_ord(1:n1,    n1+1:end);
E22 = E_ord(n1+1:end, n1+1:end);
A11 = A_ord(1:n1,    1:n1);  
A12 = A_ord(1:n1,    n1+1:end);
A22 = A_ord(n1+1:end, n1+1:end);

% --- Step 6: Solve Sylvester equations ---
[X, Y] = slgesg(E11, A11, -E22, -A22, -E12, -A12, [1, 1], 0);

% --- Step 7: Construct transformation matrices ---
P1 = [eye(n1), Y; zeros(n2, n1), eye(n2)];
Q1 = [eye(n1), X; zeros(n2, n1), eye(n2)];
E_slgs = P1 * E_ord * Q1; 
A_slgs = P1 * A_ord * Q1; 

% --- Step 8: Final transformation ---
P2    = [inv(E11), zeros(n1,n2); zeros(n2,n1), inv(A22)];
S_out = P2 * P1 * Q_ord;
P_out = Z_ord * Q1;
end
