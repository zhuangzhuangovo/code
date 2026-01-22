function is_R_observable = check_R_observability(E, A, C)
    % CHECK_R_OBSERVABILITY Checks R-observability of a descriptor system
    %
    % Inputs:
    %   E : n×n descriptor matrix (may be singular)
    %   A : n×n state matrix
    %   C : p×n output matrix
    %
    % Output:
    %   is_R_observable : Boolean, true if the system is R-observable

    n = size(A, 1);
    tol = 1e-10;  % Numerical tolerance for rank computation

    % --- Condition: rank([sE - A; C]) = n for all finite eigenvalues s ---
    % Compute generalized eigenvalues s of the pencil (A,E)
    s = eig(A, E);
    finite_s = s(isfinite(s));  % Filter out infinite eigenvalues

    % Check rank condition for each finite eigenvalue
    for k = 1:length(finite_s)
        sk = finite_s(k);
        M = [sk*E - A; C];  % Vertical concatenation
        if rank(M, tol) < n
            is_R_observable = false;
            disp(['System is NOT R-observable: rank condition failed at s = ', num2str(sk)]);
            return;
        end
    end

    is_R_observable = true;
    disp('System is R-observable!');
end