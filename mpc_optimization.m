% ========== MPC Optimization Function ==========
function [x_hat_MPC, u_hat_MPC] = mpc_optimization(E, A, B, C, D, Q, R, x0, ys, us, L)
% MPC_OPTIMIZATION Solve MPC for descriptor (singular) systems
%
% Inputs:
%   E, A, B, C, D : state-space matrices
%   Q, R          : output/input weight matrices
%   x0            : initial state
%   ys            : output setpoint
%   us            : input setpoint
%   L             : prediction horizon
%
% Outputs:
%   x_hat_MPC     : optimized state trajectory (n x L+1)
%   u_hat_MPC     : optimized input trajectory (m x L)

n = size(A,2); % state dimension
m = size(B,2); % input dimension


cvx_begin
cvx_solver mosek

% Decision variables
variable u_hat_MPC(m, L)
variable x_hat_MPC(n, L+1)

% Initial condition
x_hat_MPC(:,1) == x0;

% Objective function
obj = 0;

for k = 1:L
    % State update for descriptor system: E*x_{k+1} = A*x_k + B*u_k
    E * x_hat_MPC(:, k+1) == A * x_hat_MPC(:, k) + B * u_hat_MPC(:, k);
    
    % Predicted output
    y_hat = C * x_hat_MPC(:, k) + D * u_hat_MPC(:, k);
    
    % Stage cost
    obj = obj + (y_hat - ys)'*Q*(y_hat - ys) + (u_hat_MPC(:,k) - us)'*R*(u_hat_MPC(:,k) - us);
end

minimize(obj)

% Input constraints
for k = 1:L
    u_hat_MPC(:, k) >= -1;
    u_hat_MPC(:, k) <= 1;
end

cvx_end

% Warning if CVX did not solve
if ~strcmp(cvx_status, 'Solved')
    warning('MPC optimization not solved. CVX status: %s', cvx_status);
end

end
