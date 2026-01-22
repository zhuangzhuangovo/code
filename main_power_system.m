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

%% ==================== Regularity and (R/C) observability ====================
s = randn + 1i*randn; % Random complex number
pencil = s*E - A;

% Check if descriptor system is regular
if rank(pencil) == size(E,1)
    disp('Descriptor system is regular');
else
    disp('Descriptor system is singular');
end

% Custom functions to check controllability/observability
check_R_observability(E, A, C);
check_R_controllability(E, A, B);

%% ==================== Quasi-Weierstrass decomposition ====================
[S, P, q, r] = quasi_weierstrass(E, A, B);
E_c = S*E*P; A_c = S*A*P; B_c = S*B; C_c = C*P;

% Numerical truncation
tol = 1e-10;
E_c(abs(E_c)<tol) = 0;
A_c(abs(A_c)<tol) = 0;
B_c(abs(B_c)<tol) = 0;

% Separate differential and algebraic parts
N_c = E_c(q+1:end,q+1:end);
A_1 = A_c(1:q,1:q);
B_1 = B_c(1:q,:);
B_2 = B_c(q+1:end,:);

%% ==================== Simulation setup ====================
s = nilpotent_index(N_c); % Algebraic part nilpotent index
p = size(C,1);
m = size(B,2);
n = size(E,2);
L = 14; N = 6; T = 100; K = 200;

% Random input for open-loop simulation
u_bar = rand(m,T-s+1)*2 - 1;

% Initialize differential states
z1 = 5*rand(q,1); 
y_bar = zeros(p,T-s+1);
x_bar = zeros(n,T-s+1);

% Simulate system to generate output data
for t = 1:T-s+1
    z2 = -B_2*u_bar(:,t);           % Algebraic states
    x_bar(:,t) = P*[z1; z2];        % Full descriptor state
    y_bar(:,t) = C_c*[z1; z2];      % Output
    if t < T
        z1 = A_1*z1 + B_1*u_bar(:,t); % Update differential states
    end
end
x0 = P*[z1; z2];

%% ==================== Construct Hankel matrix ====================
H = hankel(m, p, s, L, N, T, u_bar, y_bar);

% Select data window for identification
i = 10;
u = u_bar(:,end-T+3:end);
y = y_bar(:,end-T+3:end);

[E_re,A_re,B_re,C_re,D_re,n_re,x0_id,j] = identify_descriptor_system(u, y, i);

% Propagate initial state for identified system
for t = i+j+1:T-s+1
    x0_id = pinv(E_re)*(A_re*x0_id + B_re*u_bar(:,t));
end

%% ==================== Control weights ====================
Q = 30*eye(p); % Output weight
R = eye(m);    % Input weight

%% ==================== Initialize online control variables ====================
u = u_bar(:,end-N+1:end);
y = y_bar(:,end-N+1:end);

% Discretized storage
u_dis = []; y_dis = [];
u_dis_mpc = []; y_dis_mpc = [];
u_dis_id_mpc = []; y_dis_id_mpc = [];

cvx_clear; % Ensure CVX clean

%% ==================== Online DeePC / MPC / ID+MPC ====================
t = 1;
while t < K-T+s
    % Setpoint
    if t <= 50
        ys = [3;3];
    else
        ys = [1;5];
    end
    us = zeros(m,1); % Input reference

    % DeePC optimization
    [u_hat, y_hat] = deepc_optimization(H, Q, R, ys, us, u, y, N, L, 1000,false);
    u = u_hat(:,2:N+1); 
    y = y_hat(:,2:N+1); 
    u_dis = [u_dis, u_hat(:,N+1)];
    y_dis = [y_dis, y_hat(:,N+1)];
    cvx_clear;

    % MPC
    [x_hat_MPC, u_hat_MPC] = mpc_optimization(E, A, B, C, zeros(p,m), Q, R, x0, ys, us, L);
    u_dis_mpc = [u_dis_mpc, u_hat_MPC(:,2)];
    y_dis_mpc = [y_dis_mpc, C*x_hat_MPC(:,2)];
    x0 = x_hat_MPC(:,2);
    cvx_clear;

    % ID + MPC
    [x_hat_id_MPC, u_hat_id_MPC] = mpc_optimization(E_re, A_re, B_re, C_re, D_re, Q, R, x0_id, ys, us, L);
    u_dis_id_mpc = [u_dis_id_mpc, u_hat_id_MPC(:,1)];
    y_dis_id_mpc = [y_dis_id_mpc, C_re*x_hat_id_MPC(:,1)];
    x0_id = x_hat_id_MPC(:,2);
    cvx_clear;
    t
    t = t + 1;
end

% Concatenate open-loop data
y_dis = [y_bar, y_dis]; u_dis = [u_bar, u_dis];
y_dis_mpc = [y_bar, y_dis_mpc]; u_dis_mpc = [u_bar, u_dis_mpc];
y_dis_id_mpc = [y_bar, y_dis_id_mpc]; u_dis_id_mpc = [u_bar, u_dis_id_mpc];

%% ==================== Plot output comparison ====================
%% ==================== Output comparison ====================
figure;
sgtitle('output: DeePC vs MPC vs ID-MPC', 'FontSize', 10);

time = 1:K;

% ==================== Output y1 ====================
subplot(2,1,1);
hold on;

% Determine y-axis limits
all_y1_data = [y_dis(1,:), y_dis_mpc(1,:), y_dis_id_mpc(1,:),1,3];
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
plot(time, y_dis_mpc(1,:), 'r--','LineWidth',2.5, 'DisplayName','MPC-y1');
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

all_y2_data = [y_dis(2,:), y_dis_mpc(2,:), y_dis_id_mpc(2,:), 3,5];
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
plot(time, y_dis_mpc(2,:), 'r--','LineWidth',2.5, 'DisplayName','MPC-y2');
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

    all_u_data = [u_dis(ui,:), u_dis_mpc(ui,:), u_dis_id_mpc(ui,:)];
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
    plot(time, u_dis_mpc(ui,:), 'r--','LineWidth',2.5,'DisplayName',['MPC-u',num2str(ui)]);
    plot(time, u_dis_id_mpc(ui,:), 'g:','LineWidth',2.5,'DisplayName',['ID+MPC-u',num2str(ui)]);

    % Phase separation lines
    plot([T-s+1,T-s+1],[u_min,u_max],'m--','LineWidth',1,'HandleVisibility','off');
    plot([T-s+1+50,T-s+1+50],[u_min,u_max],'m--','LineWidth',1,'HandleVisibility','off');

    legend('Location','northwest'); grid on;
    xlim([1,K]); ylim([u_min,u_max]);
    hold off;
end
