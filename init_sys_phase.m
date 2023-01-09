function [param, utility] = init_sys_phase()
    param.d_r = 0.4;
    param.d_c = 0.4;
    param.M = 16;
    param.N = 16;
    param.P = param.M*param.N/2;
    param.K = 2; % default 2
    param.mc_coef = [0.1, 0.4];
    param.amp_err_range = [0.5,1.5].';
    param.phase_err_range = deg2rad([-60,60].'); 
    param.min_space_theta = 2*100/param.M;
    param.min_space_phi = 2*100/param.N;
    param.theta_range = [20, 80].';
    param.phi_range = [-30, 30].';
    param.grid_theta_range = [10, 90].';
    param.grid_phi_range = [-40, 40].';
    param.grid_delta = 0.5;

    % RMSE-SNR performance
    param.SNR_range = single([-30:10:50].');
    param.sim_num = single(1000);

    save('param.mat', 'param', '-v7.3');
    utility.psi = @(m,n,theta,phi) vec(-vecH(n)*param.d_c*sin(theta)*sin(phi)-...
        vec(m)*param.d_r*cos(theta));
    utility.steer = @(theta, phi) exp(1j*2*pi*utility.psi([0:param.M-1], [0:param.N-1], theta, phi));
 
end