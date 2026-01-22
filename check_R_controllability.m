function is_R_controllable = check_R_controllability(E, A, B)
    % CHECK_R_CONTROLLABILITY Checks R-controllability of a descriptor system
    %
    % Inputs:
    %   E : n×n descriptor matrix (may be singular)
    %   A : n×n state matrix
    %   B : n×m input matrix
    %
    % Output:
    %   is_R_controllable : Boolean, true if the system is R-controllable

    n = size(A, 1);
    tol = 1e-10;  % Numerical tolerance for rank computation

    % --- Condition: rank([sE - A, B]) = n for all finite eigenvalues s ---
    % Compute generalized eigenvalues s of the pencil (A,E)
    s = eig(A, E);
    finite_s = s(isfinite(s));  % Filter out infinite eigenvalues

    % Check rank condition for each finite eigenvalue
    for k = 1:length(finite_s)
        sk = finite_s(k);
        M = [sk*E - A, B];  % Horizontal concatenation
        if rank(M, tol) < n
            is_R_controllable = false;
            disp(['System is NOT R-controllable: rank condition failed at s = ', num2str(sk)]);
            return;
        end
    end

    is_R_controllable = true;
    disp('System is R-controllable!');
end