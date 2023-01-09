# Simulation Tips

## 1. Train the network
1. **main_generate_data_train.m** generate the data for traning. The system parameters are saved in the file _param.mat_, the training data is saved in the file _data_set.mat_, and the valid data is saved in the file _valid_set.mat_.
2. **train.py** train the network, the loss results are saved in the file _loss.npz_, and the trained network is saved in the file _net.pkl_。

## 2. Test the performance
### 2.1 RMSE-SNR performance
1. **main_generate_data_SNR.m** generate the data to test the RMSE-SNR performance, and the data is saved in the file _data_SNR.mat_.
2. **main_SNR.py** is used to get the denoised signal, which is saved in the file _denoise_signal.mat_.
3. **main_SNR_RMSE.m**: show the performance of different algorithms. The performance of different methods is obtained as
    - get_SNR_RMSE_performance('fft'): fft method
    - get_SNR_RMSE_performance('omp'): omp method
    - get_SNR_RMSE_performance('fft-denoise'): fft method with the proposed network
    - get_SNR_RMSE_performance('omp-denosie'): omp method with the proposed network

### 2.2 RMSE-SNR performance (change RIS amp errors)
1. **main_generate_data_SNR_amp.m** generate the data to test the RMSE-SNR performance, and the data is saved in the file _data_SNR.mat_.
2. **main_SNR.py** is used to get the denoised signal, which is saved in the file _denoise_signal.mat_.
3. **main_SNR_RMSE_amp.m**: show the performance of different algorithms. The performance of different methods is obtained as
    - get_SNR_RMSE_performance('fft'): fft method
    - get_SNR_RMSE_performance('omp'): omp method
    - get_SNR_RMSE_performance('fft-denoise'): fft method with the proposed network
    - get_SNR_RMSE_performance('omp-denosie'): omp method with the proposed network

### 2.3 RMSE-SNR performance (change RIS phase errors)
1. **main_generate_data_SNR_phase.m** generate the data to test the RMSE-SNR performance, and the data is saved in the file _data_SNR.mat_.
2. **main_SNR.py** is used to get the denoised signal, which is saved in the file _denoise_signal.mat_.
3. **main_SNR_RMSE_phase.m**: show the performance of different algorithms. The performance of different methods is obtained as
    - get_SNR_RMSE_performance('fft'): fft method
    - get_SNR_RMSE_performance('omp'): omp method
    - get_SNR_RMSE_performance('fft-denoise'): fft method with the proposed network
    - get_SNR_RMSE_performance('omp-denosie'): omp method with the proposed network

### 2.3 RMSE-SNR performance (change mutual coupling)
1. **main_generate_data_SNR_mc.m** generate the data to test the RMSE-SNR performance, and the data is saved in the file _data_SNR.mat_.
2. **main_SNR.py** is used to get the denoised signal, which is saved in the file _denoise_signal.mat_.
3. **main_SNR_RMSE_mc.m**: show the performance of different algorithms. The performance of different methods is obtained as
    - get_SNR_RMSE_performance('fft'): fft method
    - get_SNR_RMSE_performance('omp'): omp method
    - get_SNR_RMSE_performance('fft-denoise'): fft method with the proposed network
    - get_SNR_RMSE_performance('omp-denosie'): omp method with the proposed network

