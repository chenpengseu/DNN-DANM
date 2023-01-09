function [recv, B, G, C, recv_perfect] = get_recv_signal(param, utility, target_theta, target_phi, SNR_dB)
    % steering vectors
    A = zeros(param.M*param.N, param.K);
    for idx = 1:param.K
        A(:, idx) = utility.steer(deg2rad(target_theta(idx)), ...
            deg2rad(target_phi(idx)));
    end

    % target signal
    s = exp(1j*rand(param.K,1)*2*pi);

    % generate matrix G (measurement)
    G_file = load('G.mat');
    G = G_file.G;
    if norm(vec(size(G)-[param.P, size(A, 1)]))>0
        G = rand(param.P, size(A, 1));
        G(G>=0.5) = 1;
        G(G<0.5) = -1;
        save('G.mat', 'G');
        G_file = load('G.mat');
        G = G_file.G;
    end
    % if true
    %     G = rand(param.P, size(A, 1));
    %     G(G>=0.5) = 1;
    %     G(G<0.5) = -1;
    %     save('G.mat', 'G')
    %     keyboard
    % else
    %     load('G.mat');
    % end
    
    % signal power=norm(G*A*s)^2
    s_pow = param.K*param.N*param.M*param.P;

    % generate noise
    noise_pow = s_pow/db2pow(SNR_dB);
    noise = randn(param.P, 1)+1j*randn(param.P, 1);
    noise = noise/norm(noise)*sqrt(noise_pow);

    % generate matrix B (relfection error)
    B = G; 
    idx_tmp = find(B<1);
    B(idx_tmp) = (rand(length(idx_tmp), 1)*(param.amp_err_range(2)-param.amp_err_range(1))+param.amp_err_range(1))...
        .*exp(1j*(rand(length(idx_tmp), 1)*(param.phase_err_range(2)-param.phase_err_range(1))+param.phase_err_range(1)));

    % generate the mutual coupling  
    C = get_mutual_coupling(param.M,param.N,param.mc_coef);
    
    % generate received signal
    recv = (B.*G)*C*A*s+noise;
    recv_perfect = G * A * s;
end
