import numpy as np
import torch
import scipy.io as io
import h5py
import risdoa
import torch.utils.data as data_utils
import argparse
import matplotlib.pyplot as plt
import scipy.io as scio

class Param():
    pass

if __name__ =='__main__':

    param = Param()
    # system parameters
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')
    param.batch_size = 64

    param_file = h5py.File('param.mat')
    param.K = param_file['param']['K'][0,0].astype('int')
    param.M = param_file['param']['M'][0,0].astype('int')
    param.N = param_file['param']['N'][0,0].astype('int')
    param.P = param_file['param']['P'][0,0].astype('int')
    param.amp_err_range = np.array(param_file['param']['amp_err_range'][0]).astype('float32')
    param.d_c = param_file['param']['d_c'][0,0].astype('float32')
    param.d_r = param_file['param']['d_r'][0,0].astype('float32')
    param.grid_delta = param_file['param']['grid_delta'][0,0].astype('float32')
    param.grid_phi_range = np.array(param_file['param']['grid_phi_range'][0]).astype('float32')
    param.grid_theta_range = np.array(param_file['param']['grid_theta_range'][0]).astype('float32')
    param.mc_coef = param_file['param']['mc_coef'][0,0].astype('float32')
    param.min_space_phi = param_file['param']['min_space_phi'][0,0].astype('float32')
    param.min_space_theta = param_file['param']['min_space_theta'][0,0].astype('float32')
    param.phase_err_range = np.array(param_file['param']['phase_err_range'][0]).astype('float32')
    param.phi_range = np.array(param_file['param']['phi_range'][0]).astype('float32')
    param.theta_range = np.array(param_file['param']['phi_range'][0]).astype('float32')
    if torch.cuda.is_available():
        param.use_cuda = True
    else:
        param.use_cuda = False

    # read data file and generate the dataset
    data_set_file = h5py.File('data_set.mat')
    recv_test_real = np.array(data_set_file['recv_test']['real'][0]).astype('float32')
    recv_test_imag = np.array(data_set_file['recv_test']['imag'][0]).astype('float32')
    dic_mat = np.array(data_set_file['dic_mat']['real']).astype('float32')+1j*np.array(data_set_file['dic_mat']['imag']).astype('float32')
    recv_set_real = np.array(data_set_file['recv_set']['real'])
    recv_set_imag = np.array(data_set_file['recv_set']['imag'])
    recv_perfect_set_real = np.array(data_set_file['recv_perfect_set']['real'])
    recv_perfect_set_imag = np.array(data_set_file['recv_perfect_set']['imag'])
    data_num = recv_set_real.shape[0]
    measure_num = recv_set_real.shape[1]
    recv_set = np.zeros((data_num, 2, measure_num)).astype('float32')
    recv_perfect_set = np.zeros((data_num, 2, measure_num)).astype('float32')
    # ref_sp_set = np.array(data_set_file['ref_sp_set']).astype('float32').reshape(data_num,-1)
    sp_grid_phi = np.array(data_set_file['sp_grid_phi'])[0].astype('float32')
    sp_grid_theta = np.array(data_set_file['sp_grid_theta'])[0].astype('float32')
    SNR_set = np.array(data_set_file['SNR_set'])[0].astype('float32')
    target_ang_set = np.array(data_set_file['target_ang_set']).astype('float32')
    for idx_data in range(data_num):
        recv_set[idx_data, 0] = recv_set_real[idx_data].astype('float32')
        recv_set[idx_data, 1] = recv_set_imag[idx_data].astype('float32')
        recv_perfect_set[idx_data, 0] = recv_perfect_set_real[idx_data].astype('float32')
        recv_perfect_set[idx_data, 1] = recv_perfect_set_imag[idx_data].astype('float32')
    recv_set = torch.from_numpy(recv_set).float()
    recv_perfect_set = torch.from_numpy(recv_perfect_set).float()
    target_ang_set = torch.from_numpy(target_ang_set).float()
    # ref_sp_set = torch.from_numpy(ref_sp_set).float()
    dataset = data_utils.TensorDataset(recv_set, recv_perfect_set)
    recv_test = np.zeros((2, recv_test_real.size))
    recv_test[0] = recv_test_real
    recv_test[1] = recv_test_imag
    recv_test = torch.from_numpy(recv_test).float()
    train_loader = data_utils.DataLoader(dataset, batch_size=param.batch_size, shuffle=True)

    # read data file and generate the dataset
    data_set_file = h5py.File('valid_set.mat')
    recv_set_real = np.array(data_set_file['recv_set']['real'])
    recv_set_imag = np.array(data_set_file['recv_set']['imag'])
    recv_perfect_set_real = np.array(data_set_file['recv_perfect_set']['real'])
    recv_perfect_set_imag = np.array(data_set_file['recv_perfect_set']['imag'])
    data_num = recv_set_real.shape[0]
    measure_num = recv_set_real.shape[1]
    recv_set = np.zeros((data_num, 2, measure_num)).astype('float32')
    recv_perfect_set = np.zeros((data_num, 2, measure_num)).astype('float32')
    # ref_sp_set = np.array(data_set_file['ref_sp_set']).astype('float32').reshape(data_num, -1)
    SNR_set = np.array(data_set_file['SNR_set'])[0].astype('float32')
    target_ang_set = np.array(data_set_file['target_ang_set']).astype('float32')
    for idx_data in range(data_num):
        recv_set[idx_data, 0] = recv_set_real[idx_data].astype('float32')
        recv_set[idx_data, 1] = recv_set_imag[idx_data].astype('float32')
        recv_perfect_set[idx_data, 0] = recv_perfect_set_real[idx_data].astype('float32')
        recv_perfect_set[idx_data, 1] = recv_perfect_set_imag[idx_data].astype('float32')
    recv_set = torch.from_numpy(recv_set).float()
    recv_perfect_set = torch.from_numpy(recv_perfect_set).float()
    target_ang_set = torch.from_numpy(target_ang_set).float()
    # ref_sp_set = torch.from_numpy(ref_sp_set).float()
    validset = data_utils.TensorDataset(recv_set, recv_perfect_set)
    valid_loader = data_utils.DataLoader(dataset, batch_size=param.batch_size)

    is_continue_train = True

    if is_continue_train:
        loss_file = np.load('loss.npz')
        loss_train = loss_file['arr_0']
        loss_valid = loss_file['arr_1']
        if param.use_cuda:
            net = torch.load('net.pkl')
        else:
            net = torch.load('net.pkl', map_location=torch.device('cpu'))
    else:
        net = risdoa.ris_doa_module(measure_num=param.P, element_num=param.M*param.N)
        loss_train = np.empty([0])
        loss_valid = np.empty([0])

    if param.use_cuda:
        net.cuda()
        recv_test = recv_test.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss(reduction='sum')
    epoch_num = 10000
    for epoch in range(0, epoch_num):
        net,loss_train_tmp,loss_valid_tmp = risdoa.train_net(param, net, optimizer, criterion, train_loader, valid_loader, dic_mat, epoch)
        loss_train = np.append(loss_train, loss_train_tmp)
        loss_valid = np.append(loss_valid, loss_valid_tmp)

        if epoch%10==0:
            np.savez('loss.npz', loss_train, loss_valid)
            torch.save(net, 'net.pkl')

            plt.figure(1)
            plt.semilogy(loss_train)
            plt.semilogy(loss_valid)
            plt.show()

            # output_net = net(recv_test.view(1, param.P * 2)).detach().view(2, -1)
            # a_tmp = torch.from_numpy(np.real(dic_mat)).float()
            # b_tmp = torch.from_numpy(np.imag(dic_mat)).float()
            # if param.use_cuda:
            #     a_tmp,b_tmp = a_tmp.cuda(), b_tmp.cuda()
            # u_tmp = output_net[0, :].unsqueeze(0).T
            # v_tmp = output_net[1, :].unsqueeze(0).T
            # out_tmp_real = torch.mm(a_tmp, u_tmp) + torch.mm(b_tmp, v_tmp)
            # out_tmp_imag = torch.mm(a_tmp, v_tmp) - torch.mm(b_tmp, u_tmp)
            # out_sp = (out_tmp_real * out_tmp_real + out_tmp_imag * out_tmp_imag).T
            # # out_sp = 1 / out_sp
            # out_sp = out_sp.cpu().numpy().reshape(sp_grid_theta.size, -1)
            # scio.savemat('AA.mat', {'AA': out_sp})

    plt.figure(1)
    plt.semilogy(loss_train)
    plt.show()