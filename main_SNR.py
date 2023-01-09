import numpy as np
import torch
import scipy.io as io
import h5py
import risdoa
import torch.utils.data as data_utils
import argparse
import matplotlib.pyplot as plt
import scipy.io as scio

if __name__ =='__main__':
    data_SNR_file = h5py.File('data_SNR.mat')
    recv_real = np.array(data_SNR_file['data_SNR']['recv']['real'])
    recv_imag = np.array(data_SNR_file['data_SNR']['recv']['imag'])
    data_num = recv_real.shape[0]
    measure_num = recv_real.shape[1]
    recv_set = np.zeros((data_num, 2, measure_num)).astype('float32')
    for idx_data in range(data_num):
        recv_set[idx_data, 0] = recv_real[idx_data].astype('float32')
        recv_set[idx_data, 1] = recv_imag[idx_data].astype('float32')
    recv_set = torch.from_numpy(recv_set).float()

    if torch.cuda.is_available():
        net = torch.load('net.pkl')
        net.cuda()
        recv_set = recv_set.cuda()
    else:
        net = torch.load('net.pkl', map_location=torch.device('cpu'))

    output_net = net(recv_set).view(recv_set.shape[0], 2, -1)
    denoise_signal = output_net.cpu().detach().numpy()
    scio.savemat('denoise_signal.mat', {'denoise_signal': denoise_signal})
    print('Good!')