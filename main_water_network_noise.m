clc; clear all;

%% ==================== Load EPANET Network Data ====================
epanetData = readEPANETFile('Net3.inp');

junction_infoMatrix = epanetData.JUNCTIONS;   % Junction data
tanks_infoMatrix = epanetData.TANKS;          % Tank data
pipe_infoMatrix = epanetData.PIPES;           % Pipe data
reservoirs_infoMatrix = epanetData.RESERVOIRS; % Reservoir data
pump_infoMatrix = epanetData.PUMPS;           % Pump data

%% ==================== Initialize Tank Cross-Section Area ====================
csa = zeros(numel(tanks_infoMatrix), numel(tanks_infoMatrix));
for i = 1:numel(tanks_infoMatrix)
    diameter = tanks_infoMatrix(i).Diameter * 0.0254; % meters
    r = diameter / 2;                                  % radius
    csa(i,i) = pi * r^2;                               % cross-section area
end

M = blkdiag(1, csa);        % Build block-diagonal
E = blkdiag(1, M, zeros(92)); % Descriptor matrix

convertToChar = @(id) num2str(id); % Helper function to unify IDs as string

%% ==================== Step 1: Build Node Index Map ====================
% Node order: Reservoirs -> Tanks -> Pumps (60,61) -> Other Junctions
node_order = cell(97,1);
index_map = containers.Map('KeyType','char','ValueType','double');
current_idx = 1;

% 1) Add reservoirs
for k = 1:numel(reservoirs_infoMatrix)
    node_id = reservoirs_infoMatrix(k).ID;
    node_order{current_idx} = node_id;
    index_map(node_id) = current_idx;
    current_idx = current_idx + 1;
end

% 2) Add tanks
for k = 1:numel(tanks_infoMatrix)
    node_id = convertToChar(tanks_infoMatrix(k).ID);
    node_order{current_idx} = node_id;
    index_map(node_id) = current_idx;
    current_idx = current_idx + 1;
end

% 3) Add pumps (60,61)
pump_ids = [60,61];
for k = 1:numel(pump_ids)
    node_id = convertToChar(pump_ids(k));
    found = false;
    for m = 1:numel(junction_infoMatrix)
        if junction_infoMatrix(m).ID == pump_ids(k)
            node_order{current_idx} = node_id;
            index_map(node_id) = current_idx;
            current_idx = current_idx + 1;
            found = true; break;
        end
    end
    if ~found
        warning('Pump node %s not found!', node_id);
    end
end

% 4) Add remaining junctions (exclude pumps)
for k = 1:numel(junction_infoMatrix)
    raw_id = junction_infoMatrix(k).ID;
    if ~ismember(raw_id, pump_ids)
        node_id = convertToChar(raw_id);
        node_order{current_idx} = node_id;
        index_map(node_id) = current_idx;
        current_idx = current_idx + 1;
    end
end

%% ==================== Step 2: Build Conductance Matrix ====================
n_res = numel(reservoirs_infoMatrix);
n_junc = numel(junction_infoMatrix);
n_tank = numel(tanks_infoMatrix);
n_total = n_res + n_tank + n_junc;

A = zeros(n_total, n_total);

% Pipes
for k = 1:numel(pipe_infoMatrix)
    from_id = num2str(pipe_infoMatrix(k).Node1);
    to_id   = num2str(pipe_infoMatrix(k).Node2);
    if ~isKey(index_map, from_id) || ~isKey(index_map, to_id)
        warning('Pipe %d connects invalid nodes: %s -> %s', k, from_id, to_id);
        continue;
    end
    i = index_map(from_id); j = index_map(to_id);
    d = pipe_infoMatrix(k).Diameter * 0.0254;
    l = pipe_infoMatrix(k).Length * 0.3048;
    G = discretized(d,l); % User-defined function
    A(i,i) = A(i,i) - G; A(i,j) = A(i,j) + G;
    A(j,j) = A(j,j) - G; A(j,i) = A(j,i) + G;
end

% Pumps (randomized for demo)
for k = 1:numel(pump_infoMatrix)
    from_id = num2str(pump_infoMatrix(k).Node1);
    to_id   = num2str(pump_infoMatrix(k).Node2);
    if ~isKey(index_map, from_id) || ~isKey(index_map, to_id)
        warning('Pump %d connects invalid nodes: %s -> %s', k, from_id, to_id);
        continue;
    end
    i = index_map(from_id); j = index_map(to_id);
    lenPump = 100*rand*0.3048; dPump = 10*rand*0.0254;
    G = discretized(dPump, lenPump);
    A(i,i) = A(i,i) - G; A(i,j) = A(i,j) + G;
    A(j,j) = A(j,j) - G; A(j,i) = A(j,i) + G;
end

%% ==================== Step 3: Build Input Matrices ====================
A(1:2,:) = 0; % Reservoirs are constant
B1 = [0; zeros(n_total-1,1)];
B2 = [zeros(n_res+n_tank,1); 1000; zeros(n_junc-1,1)];
B = [B1, B2, zeros(n_total,1)];
I = eye(n_total);

dt = 0.1;
A = A*dt + E; % Discretize
B = B*dt;
C = I([2,5],:); % Output matrix
D = zeros(4,3);

%% ==================== Step 4: System Properties & Quasi-Weierstrass ====================
check_R_controllability(E,A,B); % Check R-controllability
check_R_observability(E,A,C);  % Check R-observability

lambda = 1;
if rank([lambda*E-A, B])==4, disp('System is R-controllable'); end
if det(lambda*E - A)==0, disp('System is regular'); end

[S,P,q,r] = quasi_weierstrass(E,A,B);
E_c = S*E*P; A_c = S*A*P; B_c = S*B; C_c = C*P;

tol = 1e-10; % Numerical truncation
E_c(abs(E_c)<tol)=0; A_c(abs(A_c)<tol)=0; B_c(abs(B_c)<tol)=0;

N_c = E_c(q+1:end,q+1:end);
A_1 = A_c(1:q,1:q); B_1 = B_c(1:q,:); B_2 = B_c(q+1:end,:);

s = nilpotent_index(N_c); % Nilpotent index
p = size(C,1); m = size(B,2);n = size(E,2);
L = 12; N = 5; T = 100; K = 200;

%% ==================== Step 5: Generate Reference Data ====================
u_bar = rand(m,T-s+1)*2 - 1; % Random input
z1 = 5*rand(q,1);        % Differential state
y_bar = zeros(p,T-s+1); x_bar = zeros(n,T-s+1);

for t = 1:T-s+1 
    z2= -B_2 * u_bar(:, t); 
    x_bar(:,t) = P * [z1;z2]; 
    y = C_c * [z1; z2]; 
    SNR = 20; 
    noise_power = 10^(-SNR/10) * var(y); 
    measurement_noise = sqrt(noise_power) * randn(size(y)); 
    y_bar(:, t) = y + measurement_noise; 
    if t < T-s+1 
        z1 = A_1 * z1 + B_1 * u_bar(:, t); 
    end 
end
x0 = P*[z1; z2];

H = hankel(m,p,s,L,N,T,u_bar,y_bar); % Construct Hankel

%% ==================== Step 6: Identify System ====================
i = 10;
u = u_bar(:,end-T+3:end); y = y_bar(:,end-T+3:end);
[E_re,A_re,B_re,C_re,D_re,n_re,x0_id,j] = identify_descriptor_system_v2(u,y,i,1000,0.015);

for t = i+j+1:T-s+1
    x0_id = pinv(E_re)*(A_re*x0_id + B_re*u_bar(:,t));
end

Q = 10*eye(p); R = eye(m); % Weights

%% ==================== Step 7: Control Loop ====================
u = u_bar(:,end-N+1:end);
y = y_bar(:,end-N+1:end);
u_dis = []; y_dis = []; 
u_dis_id_mpc = []; y_dis_id_mpc = [];
cvx_clear;

t=1;
while t < K-T+s
    ys = (t<=50)*[3;3] + (t>50)*[1;5]; % Setpoint
    us = zeros(m,1);

    % DeePC
    [u_hat,y_hat] = deepc_optimization(H,Q,R,ys,us,u,y,N,L,1000,true);
    u = u_hat(:,2:N+1); y = y_hat(:,2:N+1);
    u_dis = [u_dis, u_hat(:,N+1)]; y_dis = [y_dis, y_hat(:,N+1)];
    cvx_clear;

    % ID-MPC
    [x_hat_id_MPC,u_hat_id_MPC] = mpc_optimization(E_re,A_re,B_re,C_re,D_re,Q,R,x0_id,ys,us,L);
    u_dis_id_mpc = [u_dis_id_mpc,u_hat_id_MPC(:,1)];
    y_dis_id_mpc = [y_dis_id_mpc,C_re*x_hat_id_MPC(:,1)];
    x0_id = x_hat_id_MPC(:,2);
    cvx_clear;
    t
    t = t+1;
end

y_dis = [y_bar, y_dis]; u_dis = [u_bar, u_dis];
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
