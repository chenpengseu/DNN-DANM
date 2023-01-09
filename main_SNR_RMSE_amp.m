clear all; close all;
% get_SNR_RMSE_amp('fft');
% get_SNR_RMSE_amp('omp');
% get_SNR_RMSE_amp('fft-denoise');
% get_SNR_RMSE_amp('omp-denoise');
% get_SNR_RMSE_amp('omp-denoise-anm');
% get_SNR_RMSE_amp('omp-denoise-danm');
get_SNR_RMSE_amp('prony-denoise-danm');

SNR_RMSE_file = load('SNR_RMSE.mat');
SNR_RMSE = SNR_RMSE_file.SNR_RMSE;
figure;
legend_str = {'omp', 'fft', 'omp-denoise', 'fft-denoise','omp-denoise-anm', 'omp-denoise-danm', 'prony-denoise-danm'};
style_str = {'^:','s:','^-','s-', '^--', 'v-', 'o-'};
for idx_fig = 1:length(SNR_RMSE)
    semilogy(SNR_RMSE{idx_fig}(:, 1), smoothdata(SNR_RMSE{idx_fig}(:, 2),'gaussian', 3), style_str{idx_fig}, 'LineWidth', 2, 'MarkerSize', 10);
    hold on;
end
legend(legend_str);
set(get(gca, 'XLabel'), 'String', 'SNR (dB)');
set(get(gca, 'YLabel'), 'String', 'RMSE (deg)');
grid on;
