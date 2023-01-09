function main_generate_data_SNR() 
    [param, utility] = init_sys();

    file_str = 'data_SNR.mat';

    [param_data, data_SNR] = generate_data_SNR(param, utility, file_str);
end
