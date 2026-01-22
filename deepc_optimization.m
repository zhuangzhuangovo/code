function [u_hat, y_hat, alpha_opt, delta1] = deepc_optimization(H, Q, R, ys, us, u_current, y_current, N, L, penalty_weight, noise_flag)
% DEEPC_OPTIMIZATION Solve the DeePC predictive control problem
%
% Inputs:
%   H             : Hankel matrix
%   Q             : output weight matrix (p x p)
%   R             : input weight matrix (m x m)
%   ys            : output setpoint (p x 1)
%   us            : input setpoint (m x 1)
%   u_current     : current input trajectory (m x N)
%   y_current     : current output trajectory (p x N)
%   N             : length of historical window
%   L             : prediction horizon
%   penalty_weight: terminal constraint relaxation weight
%   noise_flag    : true if measurement noise should be considered
%
% Outputs:
%   u_hat     : optimized input sequence (m x (N+L))
%   y_hat     : predicted output sequence (p x (N+L))
%   alpha_opt : Hankel matrix coefficients
%   delta1    : terminal slack variable

[m, ~] = size(u_current);
[p, ~] = size(y_current);

% --- CVX optimization ---
cvx_begin
cvx_solver mosek

% Decision variables
variable u_hat(m, N+L)
variable y_hat(p, N+L)
variable alpha_opt(size(H,2),1)
variable delta1((m+p)*N,1)
if noise_flag
    variable sigma_y(p*N,1)  % slack for noise
end

% --- Objective function ---
obj = 0;
for k = N+1:N+L
    obj = obj + (y_hat(:,k) - ys)'*Q*(y_hat(:,k) - ys) + u_hat(:,k)'*R*u_hat(:,k);
end
obj = obj + penalty_weight*norm(delta1, 'fro');
if noise_flag
    obj = obj + 1*(norm(alpha_opt,'fro') + 1000 * norm(sigma_y,'fro'));
end
minimize(obj)

% --- Constraints ---

% Dynamics via Hankel
if noise_flag
    [reshape(u_hat,[],1); reshape(y_hat,[],1)] + [zeros(m*N,1); sigma_y; zeros((m+p)*L,1)] == H * alpha_opt;
else
    [reshape(u_hat,[],1); reshape(y_hat,[],1)] == H * alpha_opt;
end

% Initial condition
[u_current(:); y_current(:)] == [reshape(u_hat(:,1:N),[],1); reshape(y_hat(:,1:N),[],1)];

% Terminal constraints
u_terminal = u_hat(:, end-N+1:end);
y_terminal = y_hat(:, end-N+1:end);
terminal_error = [reshape(u_terminal - repmat(us,1,N),[],1); reshape(y_terminal - repmat(ys,1,N),[],1)];
terminal_error + delta1 == 0;

% Input bounds
for k = N+1:N+L
    u_hat(:,k) >= -1;
    u_hat(:,k) <= 1;
end

cvx_end

% Check solution status
if ~strcmp(cvx_status, 'Solved')
    warning('Optimization not solved successfully. CVX status: %s', cvx_status);
end

end
