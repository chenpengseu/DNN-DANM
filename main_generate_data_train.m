% generate the training data
function [recv_set, target_ang_set, SNR_set, ref_sp_set, sp_grid_theta, sp_grid_phi] = main_generate_data_train(data_num)
    [param, utility] = init_sys(); 
    SNR_range = [20,50].';

    % generate grids
    sp_grid_theta = [param.grid_theta_range(1):param.grid_delta:param.grid_theta_range(2)].';
    sp_grid_phi = [param.grid_phi_range(1):param.grid_delta:param.grid_phi_range(2)].';
    [sp_grid_theta_mat, sp_grid_phi_mat] = meshgrid(sp_grid_theta, sp_grid_phi);
    sp_grid = zeros(size(sp_grid_theta_mat,1),size(sp_grid_theta_mat,2),2);
    sp_grid(:, :, 1)=sp_grid_theta_mat;
    sp_grid(:, :, 2)=sp_grid_phi_mat;
    dic_mat = zeros(param.M*param.N, length(sp_grid_theta)*length(sp_grid_phi));
    for idx_theta = 1:length(sp_grid_theta)
        for idx_phi = 1:length(sp_grid_phi)
            tmp = utility.steer(deg2rad(sp_grid_theta(idx_theta)), deg2rad(sp_grid_phi(idx_phi)));
            dic_mat(:, idx_phi+length(sp_grid_phi)*(idx_theta-1)) = tmp;
        end
    end 
    dic_mat = single(dic_mat);

    %% generate test signal
    % generate the target positions
    [target_theta, target_phi] = get_target_pos(param.K, param.theta_range, param.phi_range,...
                                    param.min_space_theta, param.min_space_phi);
    target_ang = [target_theta, target_phi].';

    % generate the received signal
    SNR_dB = 30;
    [recv_test, B, G, C, recv_perfect_test] = get_recv_signal(param, utility, target_theta, target_phi, SNR_dB);
    recv_test = single(recv_test);
    recv_perfect_test = single(recv_perfect_test);

    % % generate reference spectrum
    % ref_sp = zeros(size(sp_grid,1), size(sp_grid,2));
    % sp_sigma = 10;
    % for idx_k = 1:size(target_ang, 2)
    %     target_tmp = zeros(size(sp_grid));
    %     target_tmp(:, :, 1) =  target_ang(1, idx_k);
    %     target_tmp(:, :, 2) =  target_ang(2, idx_k);
    %     ref_sp = ref_sp+exp(-sum(abs(target_tmp - sp_grid).^2, 3)/sp_sigma^2);
    % end       
    % ref_sp_test = ref_sp/max(ref_sp,[],'all');
    
    %% generate data set  
    target_ang_set=zeros(2, param.K, data_num);
    recv_set = zeros(size(recv_test,1), data_num);
    recv_perfect_set = zeros(size(recv_perfect_test, 1), data_num);
    SNR_set = zeros(data_num,1);
    % ref_sp_set = zeros(size(ref_sp_test,1), size(ref_sp_test, 2), data_num);
    for idx_set = 1:data_num
        if mod(idx_set, 100) == 0
            fprintf('%.2f %%...\n', idx_set/data_num*100);
        end
        % generate the target positions
        [target_theta, target_phi] = get_target_pos(param.K, param.theta_range, param.phi_range,...
                                        param.min_space_theta, param.min_space_phi);
        target_ang = [target_theta, target_phi].';

        % generate the received signal
        SNR_dB = rand(1)*(SNR_range(2)-SNR_range(1))+SNR_range(1);
        [recv, B, G, C, recv_perfect] = get_recv_signal(param, utility, target_theta, target_phi, SNR_dB);

        % % generate reference spectrum
        % ref_sp = zeros(size(sp_grid,1), size(sp_grid,2));
        % sp_sigma = 10;
        % for idx_k = 1:size(target_ang, 2)
        %     target_tmp = zeros(size(sp_grid));
        %     target_tmp(:, :, 1) =  target_ang(1, idx_k);
        %     target_tmp(:, :, 2) =  target_ang(2, idx_k);
        %     ref_sp = ref_sp+exp(-sum(abs(target_tmp - sp_grid).^2, 3)/sp_sigma^2);
        % end 
        % ref_sp = ref_sp/max(ref_sp,[],'all');


        target_ang_set(:, :, idx_set) = target_ang;
        recv_set(:, idx_set) = recv;
        recv_perfect_set(:, idx_set) = recv_perfect;
        SNR_set(idx_set) = SNR_dB;
        % ref_sp_set(:, :, idx_set) = ref_sp;
    end 
    % ref_sp_set = single(ref_sp_set);
    recv_set = single(recv_set);
    recv_perfect_set = single(recv_perfect_set);
    target_ang_set = single(target_ang_set);
    SNR_set = single(SNR_set);
    sp_grid_theta = single(sp_grid_theta);
    sp_grid_phi = single(sp_grid_phi);


%     figure; mesh(sp_grid_theta, sp_grid_phi, ref_sp);
%     hold on; stem3(target_ang(1,:), target_ang(2,:), ones(size(target_ang,2),1));
%     set(get(gca, 'XLabel'), 'String', '\theta');
%     set(get(gca, 'YLabel'), 'String', '\phi');
%     drawnow;

    
    save('data_set.mat', 'recv_set', 'target_ang_set', 'SNR_set', 'recv_perfect_set', ...
'sp_grid_theta', 'sp_·grid_phi','dic_mat','recv_test', 'recv_perfect_test', '-v7.3');


    %% generate valid data set  
    data_num = 64;
    target_ang_set=zeros(2, param.K, data_num);
    recv_set = zeros(size(recv_test, 1), data_num);
    recv_perfect_set = zeros(size(recv_perfect_test, 1), data_num);
    SNR_set = zeros(data_num,1);
    % ref_sp_set = zeros(size(ref_sp_test,1), size(ref_sp_test, 2), data_num);
    for idx_set = 1:data_num
        if mod(idx_set, 100) == 0
            fprintf('%.2f %%...\n', idx_set/data_num*100);
        end
        % generate the target positions
        [target_theta, target_phi] = get_target_pos(param.K, param.theta_range, param.phi_range,...
                                        param.min_space_theta, param.min_space_phi);
        target_ang = [target_theta, target_phi].';

        % generate the received signal
        SNR_dB = rand(1)*(SNR_range(2)-SNR_range(1))+SNR_range(1);
        [recv, B, G, C, recv_perfect] = get_recv_signal(param, utility, target_theta, target_phi, SNR_dB);

        % % generate reference spectrum
        % ref_sp = zeros(size(sp_grid,1), size(sp_grid,2));
        % sp_sigma = 10;
        % for idx_k = 1:size(target_ang, 2)
        %     target_tmp = zeros(size(sp_grid));
        %     target_tmp(:, :, 1) =  target_ang(1, idx_k);
        %     target_tmp(:, :, 2) =  target_ang(2, idx_k);
        %     ref_sp = ref_sp+exp(-sum(abs(target_tmp - sp_grid).^2, 3)/sp_sigma^2);
        % end 
        % ref_sp = ref_sp/max(ref_sp,[],'all');


        target_ang_set(:, :, idx_set) = target_ang;
        recv_set(:, idx_set) = recv;
        recv_perfect_set(:, idx_set) = recv_perfect;
        SNR_set(idx_set) = SNR_dB;
        % ref_sp_set(:, :, idx_set) = ref_sp;
    end 
    % ref_sp_set = single(ref_sp_set);
    recv_set = single(recv_set);
    recv_perfect_set = single(recv_perfect_set);
    target_ang_set = single(target_ang_set);
    SNR_set = single(SNR_set);
    sp_grid_theta = single(sp_grid_theta);
    sp_grid_phi = single(sp_grid_phi); 

    save('valid_set.mat', 'recv_set', 'target_ang_set', 'SNR_set', 'recv_perfect_set', '-v7.3');

end