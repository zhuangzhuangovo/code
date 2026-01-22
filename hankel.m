function H = hankel(m,p,s,L,N,T,u_bar,y_bar)

    %% ==================== Construct Input Hankel Matrix ====================
    Hu = zeros(m*(N+L), (T-L-N-s+2)); % Preallocate input Hankel matrix
    for a = 1 : m*(N+L)
        for b = 1:(T-L-N-s+2)
            idx = fix((a-1)/m) + b;   % Compute row offset
            switch mod(a, m)           % Map to appropriate input channel
                case 0
                    Hu(a, b) = u_bar(3, idx);
                case 1
                    Hu(a, b) = u_bar(1, idx);
                case 2
                    Hu(a, b) = u_bar(2, idx);
            end
        end
    end

    %% ==================== Construct Output Hankel Matrix ====================
    Hy = zeros(p*(N+L), (T-L-N-s+2)); % Preallocate output Hankel matrix
    for a = 1 : p*(N+L)
        for b = 1:(T-L-N-s+2)
            idx = fix((a-1)/p) + b;   % Compute row offset
            switch mod(a, p)           % Map to appropriate output channel
                case 0
                    Hy(a, b) = y_bar(2, idx);
                case 1
                    Hy(a, b) = y_bar(1, idx);
                % case 2
                %     Hy(a,b) = y_bar(2,idx); % Uncomment if using 3 outputs
            end
        end
    end

    %% ==================== Combine Input and Output Hankel Matrices ====================
    H = [Hu; Hy];
end
