%% ==================== Clear environment ====================
clc; clear; close all;

%% ==================== Load system matrices ====================
load('L_matrix.mat'); % Load coupling matrix L

% Mass and damping matrices
M = diag([0.125,0.034,0.016]);
D = diag([0.125,-0.068,0.048]);

% Descriptor system matrices
E = [zeros(3,3), eye(3), zeros(3,6);
     M, zeros(3,9);
     zeros(6,12)];
D1 = [-D; zeros(6,3)];
A = [eye(3), zeros(3,9);
     D1, L];

% Input matrix
B = [zeros(3,3); [1,1,0;-1,0,1;0,-1,-1]; zeros(6,3)];

% Output matrix
C = [zeros(1,2),1,zeros(1,9);
     zeros(1,3),1,zeros(1,8)];

% Time step
tau = 0.01;
A = E + tau*A;
B = tau*B;
%% ==================== Regularity check ======================
s = randn + 1i * randn;
pencil = s * E - A;

n = size(E,1);
if rank(pencil) == n
    disp('The descriptor system is regular');
else
    disp('The descriptor system is NOT regular');
end

% Structural properties
check_R_observability(E, A, C);
check_R_controllability(E, A, B);
check_C_observability(E, A, C);

%% ================= Quasi-Weierstrass form ===================
[S, P, q, r] = quasi_weierstrass(E, A, B);

E_c = S * E * P;
A_c = S * A * P;
B_c = S * B;
C_c = C * P;

% Numerical truncation
tol = 1e-10;
E_c(abs(E_c) < tol) = 0;
A_c(abs(A_c) < tol) = 0;
B_c(abs(B_c) < tol) = 0;

% Partitioned subsystems
N_c = E_c(q+1:end, q+1:end);   % Nilpotent part
A_1 = A_c(1:q, 1:q);           % Dynamic subsystem
B_1 = B_c(1:q, :);
B_2 = B_c(q+1:end, :);

s = nilpotent_index(N_c);

%% ================= Simulation parameters ====================
p = size(C,1);     % Output dimension
m = size(B,2);     % Input dimension
n = size(E,2);
L = 14;            % Prediction horizon
N = 6;             % Control horizon
T = 100;           % Data length
K = 200;           % Total simulation steps

%% ================= Data generation ==========================
u_bar = rand(m, T-s+1) * 2 - 1;

z1 = 5 * rand(q,1);    % Initial differential state

y_bar = zeros(p, T-s+1);
x_bar = zeros(n, T-s+1);

for t = 1:T-s+1
    % Algebraic state
    z2 = -B_2 * u_bar(:,t);
    x_bar(:,t) = P * [z1; z2];

    % Output with measurement noise
    y = C_c * [z1; z2];
    SNR = 20;
    noise_power = 10^(-SNR/10) * var(y);
    y_bar(:,t) = y + sqrt(noise_power) * randn(size(y));

    % Differential state update
    if t < T-s+1
        z1 = A_1 * z1 + B_1 * u_bar(:,t);
    end
end

x0 = P * [z1; z2];

%% ================= System identification ====================
H = Hankel_NL(m, p, s, N, L, T, u_bar, y_bar);

i = 8;
u = u_bar(:, end-T+3:end);
y = y_bar(:, end-T+3:end);

opts = 2e-2;
[E_re, A_re, B_re, C_re, D_re, n_re, x0_id, j] = ...
    identify_descriptor_system_v2(u, y, i, 1000, opts);

for t = i+j+1:T-s+1
    x0_id = pinv(E_re) * (A_re * x0_id + B_re * u_bar(:,t));
end

%% ================= DeePC vs ID+MPC ==========================
Q = 30 * eye(p);
R = 1 * eye(m);

u = u_bar(:, end-N+1:end);
y = y_bar(:, end-N+1:end);

u_dis = [];
y_dis = [];
u_dis_id_mpc = [];
y_dis_id_mpc = [];

cvx_clear;
t = 1;

while t < K - T + s

    % Time-varying setpoint
    if t <= 50
        ys = [3; 3];
    else
        ys = [1; 5];
    end
    us = zeros(m,1);

    % DeePC optimization
    [u_hat, y_hat] = deepc_optimization( ...
        H, Q, R, ys, us, u, y, N, L, 1000, true);

    u = u_hat(:,2:N+1);
    y = y_hat(:,2:N+1);

    u_dis = [u_dis, u_hat(:,N+1)];
    y_dis = [y_dis, y_hat(:,N+1)];

    cvx_clear;

    % Identified-model MPC
    [x_hat_id, u_hat_id] = ...
        mpc_optimization(E_re, A_re, B_re, C_re, D_re, Q, R, x0_id, ys, us, L);

    u_dis_id_mpc = [u_dis_id_mpc, u_hat_id(:,1)];
    y_dis_id_mpc = [y_dis_id_mpc, C_re * x_hat_id(:,1)];
    x0_id = x_hat_id(:,2);

    cvx_clear;
    t
    t = t + 1;
end

%% ================= Data concatenation =======================
y_dis = [y_bar, y_dis];
u_dis = [u_bar, u_dis];

y_dis_id_mpc = [y_bar, y_dis_id_mpc];
u_dis_id_mpc = [u_bar, u_dis_id_mpc];

%% ==================== Plot output comparison ====================
%% ==================== Output comparison ====================
figure;
sgtitle('output: DeePC vs MPC vs ID-MPC', 'FontSize', 10);

time = 1:K;

% ==================== Output y1 ====================
subplot(2,1,1);
hold on;

% Determine y-axis limits
all_y1_data = [y_dis(1,:), y_dis_id_mpc(1,:),1,3];
y_min = min(all_y1_data) - 0.5;
y_max = max(all_y1_data) + 0.5;

% Background for open-loop training phase
patch([1, T-s+1, T-s+1, 1], [y_min, y_min, y_max, y_max], [1,0.8,0.8], ...
      'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');

% Background for first control phase
stage1_end = min(T-s+1+50, K);
if stage1_end > T-s+1
    patch([T-s+1, stage1_end, stage1_end, T-s+1], [y_min, y_min, y_max, y_max], ...
          [0.8,0.9,1], 'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
end

% Background for second control phase
if K > stage1_end
    patch([stage1_end, K, K, stage1_end], [y_min, y_min, y_max, y_max], ...
          [0.8,1,0.8],'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
end

% Plot outputs
plot(time, y_dis(1,:), 'b-', 'LineWidth', 2.5, 'DisplayName','DeePC-y1');
plot(time, y_dis_id_mpc(1,:), 'g:','LineWidth',2.5, 'DisplayName','ID+MPC-y1');

% Plot setpoint line
plot([T-s+1, T-s+1+50, T-s+1+50, K], [3,3,1,1], 'k-', 'LineWidth', 2, 'DisplayName','Setpoint');

% Vertical lines for phase separation
plot([T-s+1, T-s+1], [y_min, y_max], 'k--','LineWidth',1,'HandleVisibility','off');
plot([T-s+1+50, T-s+1+50], [y_min, y_max], 'k--','LineWidth',1,'HandleVisibility','off');

legend('Location','northwest','NumColumns',2,'FontSize',9);
grid on;
xlim([1,K]); ylim([y_min, y_max]);
hold off;

% ==================== Output y2 ====================
subplot(2,1,2);
hold on;

all_y2_data = [y_dis(2,:), y_dis_id_mpc(2,:), 3,5];
y_min = min(all_y2_data) - 0.5;
y_max = max(all_y2_data) + 0.5;

% Training phase background
patch([1, T-s+1, T-s+1, 1], [y_min, y_min, y_max, y_max], [1,0.8,0.8], ...
      'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');

% First control phase
if stage1_end > T-s+1
    patch([T-s+1, stage1_end, stage1_end, T-s+1], [y_min, y_min, y_max, y_max], ...
          [0.8,0.9,1],'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
end

% Second control phase
if K > stage1_end
    patch([stage1_end, K, K, stage1_end], [y_min, y_min, y_max, y_max], ...
          [0.8,1,0.8],'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
end

% Plot outputs
plot(time, y_dis(2,:), 'b-', 'LineWidth', 2.5, 'DisplayName','DeePC-y2');
plot(time, y_dis_id_mpc(2,:), 'g:','LineWidth',2.5, 'DisplayName','ID+MPC-y2');

% Plot setpoint line
plot([T-s+1, T-s+1+50, T-s+1+50, K], [3,3,5,5], 'k-', 'LineWidth',2, 'DisplayName','Setpoint');

% Phase separation lines
plot([T-s+1, T-s+1], [y_min, y_max], 'k--','LineWidth',1,'HandleVisibility','off');
plot([T-s+1+50, T-s+1+50], [y_min, y_max], 'k--','LineWidth',1,'HandleVisibility','off');

legend('Location','northwest','NumColumns',2,'FontSize',9);
grid on; xlim([1,K]); ylim([y_min, y_max]);
hold off;

%% ==================== Input comparison ====================
figure; sgtitle('input: DeePC vs MPC vs ID+MPC', 'FontSize', 10);

for ui = 1:m
    subplot(m,1,ui); hold on;

    all_u_data = [u_dis(ui,:), u_dis_id_mpc(ui,:)];
    u_min = min(all_u_data) - 0.5;
    u_max = max(all_u_data) + 0.5;

    % Training phase background
    patch([1,T-s+1,T-s+1,1],[u_min,u_min,u_max,u_max],[1,0.8,0.8], ...
          'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');

    % First control phase
    if stage1_end > T-s+1
        patch([T-s+1, stage1_end, stage1_end, T-s+1],[u_min,u_min,u_max,u_max], ...
              [0.8,0.9,1],'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
    end

    % Second control phase
    if K > stage1_end
        patch([stage1_end,K,K,stage1_end],[u_min,u_min,u_max,u_max], ...
              [0.8,1,0.8],'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
    end

    % Plot inputs
    plot(time, u_dis(ui,:), 'b-', 'LineWidth',2.5,'DisplayName',['DeePC-u',num2str(ui)]);
    plot(time, u_dis_id_mpc(ui,:), 'g:','LineWidth',2.5,'DisplayName',['ID+MPC-u',num2str(ui)]);

    % Phase separation lines
    plot([T-s+1,T-s+1],[u_min,u_max],'m--','LineWidth',1,'HandleVisibility','off');
    plot([T-s+1+50,T-s+1+50],[u_min,u_max],'m--','LineWidth',1,'HandleVisibility','off');

    legend('Location','northwest'); grid on;
    xlim([1,K]); ylim([u_min,u_max]);
    hold off;
end
