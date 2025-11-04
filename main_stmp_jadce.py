import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import argparse
import torch
import numpy as np
import h5py
import time
import yaml
import matplotlib.pyplot as plt

from model.ncsn_channel_diffusion import *
from model.ncsnv2_channel_diffusion import *


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def closest_element(lst, val):
    idx = (torch.abs(lst - val)).argmin()
    return lst[idx], idx


def load_channel(filename, K, N, M):
    with h5py.File(filename, 'r') as hf:
        h_freq = hf['h_f_test'][:]

    h_freq = h_freq.transpose(3, 2, 0, 1).astype(np.float32)
    h_f = h_freq[0:K, 0, 0:M, 0:N] + 1j * h_freq[0:K, 1, 0:M, 0:N]
    h_f = h_f.transpose(0, 2, 1)  # K * N * M

    # h_f_vector = h_f.reshape(h_f.shape[0], -1)
    # cov_h = h_f_vector.conj().T @ h_f_vector / h_f.shape[0]
    # var_h = np.real(np.mean(np.diag(cov_h)))
    var_h = 1.0

    return h_f, var_h


def pilot_matrix_generation(K, N, T, P, device):
    if N * T <= K:
        Q_full = torch.fft.fft(torch.eye(K, device=device)) * torch.sqrt(torch.tensor(P))
        permutation = torch.randperm(K).to(device)
        pilot_sel_idx = permutation[:N * T]
        Q_fft = Q_full[pilot_sel_idx, :]
        Q = torch.zeros((N * T, N * K), dtype=torch.complex64, device=device)
        for n in range(N):
            Q[n::N, n::N] = Q_fft[n::N, :]
    else:
        Q_full = torch.fft.fft(torch.eye(K, device=device)) * torch.sqrt(torch.tensor(P))
        Q = torch.zeros((N * T, N * K), dtype=torch.complex64, device=device)
        Q_fft = torch.zeros((N * T, K), dtype=torch.complex64, device=device)
        for n in range(N):
            permutation = torch.randperm(K).to(device)
            pilot_sel_idx = permutation[:T]
            Q_fft_n = Q_full[pilot_sel_idx, :]
            Q[n::N, n::N] = Q_fft[n::N, :] = Q_fft_n

    return Q_fft, Q, pilot_sel_idx


def received_signal_generation(h_f, r, Q, K, N, M, T, delta, device):
    x_f = r * h_f
    X = x_f.reshape(N * K, M)
    QX = Q @ X
    Noise = (delta / torch.sqrt(torch.tensor(2.0, device=device))) * (
                torch.randn(N * T, M, device=device) + 1j * torch.randn(N * T, M, device=device))
    Y = QX + Noise

    return Y


class STMP_JADCE(object):
    def __init__(self, args, s1, s2, var_h, sigmas, active_idx, device):
        self.M = args.M
        self.N = args.N
        self.K = args.K
        self.T = args.T
        self.P = args.P
        self.lamda = args.lamda
        self.delta = args.delta
        self.var_h = var_h
        self.sigmas = sigmas
        self.device = device
        self.s1 = s1
        self.s2 = s2
        self.active_idx = active_idx

        self.beta = args.beta  # damping factor
        self.ite = args.ite
        self.ite_gaussian = args.ite_gaussian

    def run(self, Y, Q, h_f, active_idx):
        # initialization
        X_A_pri = torch.zeros((self.N * self.K, self.M), dtype=torch.complex64, device=self.device)
        v_X_A_pri = self.lamda * self.var_h * torch.ones(self.M, dtype=torch.float32, device=self.device)

        nmse_last = 0
        NMSE_B = []
        P_error = []
        for ite in range(self.ite):
            # module A: linear estimation of X
            X_B_pri, v_X_B_pri, X_A_post = self.module_A(Y, Q, X_A_pri, v_X_A_pri)  # X_A_ext, v_X_A_ext
            # damping
            if ite >= 1:
                X_B_pri = self.beta * X_B_pri + (1 - self.beta) * X_B_pri_old
                v_X_B_pri = self.beta * v_X_B_pri + (1 - self.beta) * v_X_B_pri_old
            X_B_pri_old = X_B_pri
            v_X_B_pri_old = v_X_B_pri

            # module B: denoiser of H
            X_A_pri, v_X_A_pri, X_B_post, active_est = self.module_B(X_B_pri, v_X_B_pri, ite, h_f)
            # damping
            if ite >= 1:
                X_A_pri = self.beta * X_A_pri + (1 - self.beta) * X_A_pri_old
                v_X_A_pri = self.beta * v_X_A_pri + (1 - self.beta) * v_X_A_pri_old
            X_A_pri_old = X_A_pri
            v_X_A_pri_old = v_X_A_pri

            # result B
            nmse_B, p_error = self.result_B(h_f, X_B_post.reshape((self.K, self.N, self.M)), active_idx, active_est)
            NMSE_B.append(nmse_B.item())
            P_error.append(p_error.item())
            print(f'NMSE: {10 * np.log10(nmse_B.item()):.4f} dB | P_error: {p_error.item():.8f}')

            # if torch.norm(nmse_last - nmse_B) < 1e-5:
            #     break
            nmse_last = nmse_B

        return NMSE_B, P_error

    def module_A(self, Y, Q, X_A_pri, v_X_A_pri):
        # posterior
        X_A_post = X_A_pri + (v_X_A_pri / (self.K * self.P * v_X_A_pri + self.delta ** 2)) * (Q.conj().T @ (Y - Q @ X_A_pri))
        v_X_A_post = v_X_A_pri - self.T * self.P * v_X_A_pri ** 2 / (self.K * self.P * v_X_A_pri + self.delta ** 2)

        # extrinsic
        v_X_A_ext = 1 / (1 / v_X_A_post - 1 / v_X_A_pri)
        X_A_ext = v_X_A_ext * (X_A_post / v_X_A_post - X_A_pri / v_X_A_pri)

        return X_A_ext, v_X_A_ext, X_A_post

    def module_B(self, X_B_pri, v_X_B_pri, ite, h_f):
        # posterior
        v_X_B_pri_real = v_X_B_pri / 2
        std_X_B_pri_real = torch.sqrt(torch.mean(v_X_B_pri_real))
        closest_std_X_B_pri, closest_idx = closest_element(self.sigmas, std_X_B_pri_real)

        self.s1.eval()
        self.s2.eval()
        with torch.no_grad():
            X_B_pri_real_nn = torch.real(X_B_pri.reshape((self.K, self.N, self.M)).permute(0, 2, 1))
            X_B_pri_imag_nn = torch.imag(X_B_pri.reshape((self.K, self.N, self.M)).permute(0, 2, 1))
            X_B_pri_nn = torch.cat((X_B_pri_real_nn.unsqueeze(1), X_B_pri_imag_nn.unsqueeze(1)), dim=1)

            labels = closest_idx * torch.ones(X_B_pri_nn.shape[0], device=self.device).long()

            # first order
            scores_1st_order_nn = self.s1(X_B_pri_nn, labels)
            H_B_post_nn = X_B_pri_nn + std_X_B_pri_real ** 2 * scores_1st_order_nn

            H_B_post = H_B_post_nn[:, 0, :, :] + 1j * H_B_post_nn[:, 1, :, :]
            H_B_post = H_B_post.permute(0, 2, 1)
            H_B_post = H_B_post.reshape((self.K * self.N, self.M))

            # second order
            scores_2nd_order_nn = self.s2(X_B_pri_nn, labels)
            v_H_B_post_nn = std_X_B_pri_real ** 4 * scores_2nd_order_nn + std_X_B_pri_real ** 2 * torch.ones_like(scores_2nd_order_nn)
            v_H_B_post_nn = torch.sum(v_H_B_post_nn, dim=1)
            v_H_B_post = torch.mean(v_H_B_post_nn, dim=(0, 2))
            # v_H_B_post = torch.abs(v_H_B_post)
            v_H_B_post[v_H_B_post < 1e-4] = 1e-4

        # extrinsic message of H_k
        v_H_B_ext = 1 / (1 / v_H_B_post - 1 / v_X_B_pri)
        H_B_ext = v_H_B_ext * (H_B_post / v_H_B_post - X_B_pri / v_X_B_pri)

        Phi = 1 / (1 / v_H_B_ext + 1 / v_X_B_pri)
        Mu = Phi * (1 / v_X_B_pri * X_B_pri + 1 / v_H_B_ext * H_B_ext)
        Mu = Mu.reshape((self.K, self.N, self.M))
        Phi = Phi.unsqueeze(0).unsqueeze(1)

        lamda_post, active_est = self.lambda_post_gaussian(X_B_pri, v_X_B_pri, ite)

        lamda_post = lamda_post.unsqueeze(1).unsqueeze(2)
        X_B_post = lamda_post * Mu
        v_X_B_post = torch.real(lamda_post * (Mu * Mu.conj() + Phi) - X_B_post * X_B_post.conj())
        v_X_B_post = torch.mean(v_X_B_post, dim=(0, 1))

        X_B_post = X_B_post.reshape((self.K * self.N, self.M))

        # extrinsic
        v_X_B_ext = 1 / (1 / v_X_B_post - 1 / v_X_B_pri)
        X_B_ext = v_X_B_ext * (X_B_post / v_X_B_post - X_B_pri / v_X_B_pri)

        print(f'v_X_B_pri: {torch.mean(v_X_B_pri).item():.4f} |',
              f'v_H_B_post: {torch.mean(v_H_B_post).item():.4f} |',
              f'v_H_B_ext: {torch.mean(v_H_B_ext).item():.4f} |'
              f'v_X_B_post: {torch.mean(v_X_B_post).item():.4f}  |',
              f'v_X_B_ext: {torch.mean(v_X_B_ext).item():.4f} |')
        return X_B_ext, v_X_B_ext, X_B_post, active_est

    def lambda_post_gaussian(self, X_B_pri, v_X_B_pri, ite):
        # Gaussian prior
        v_H_B_ext = self.var_h * torch.ones(self.M, device=self.device)
        H_B_ext = torch.zeros((self.K * self.N, self.M), dtype=torch.complex64, device=self.device)

        value_on_exp_1a = - 1 / v_X_B_pri * torch.norm(X_B_pri.reshape((self.K, self.N, self.M)), dim=1) ** 2
        value_on_exp_1b = 1 / (v_X_B_pri + v_H_B_ext) * torch.norm(
            (X_B_pri - H_B_ext).reshape((self.K, self.N, self.M)), dim=1) ** 2
        value_on_exp_1 = torch.sum(value_on_exp_1a + value_on_exp_1b, dim=1)

        value_on_exp_2 = torch.log(1 + v_H_B_ext / v_X_B_pri)
        value_on_exp_2 = self.N * torch.sum(value_on_exp_2)
        value_on_exp = value_on_exp_1 + value_on_exp_2
        value_on_exp[value_on_exp > 600] = 600

        lamda_post = 1 / (1 + (1 - self.lamda) / self.lamda * torch.exp(value_on_exp))
        active_est = torch.where(lamda_post > 0.5)[0]
        # print(active_est)

        return lamda_post, active_est

    def result_B(self, h_true, h_est, active_true, active_est):
        false_alarm = torch.isin(active_est, active_true)
        mis_detection = torch.isin(active_true, active_est)
        p_error = (torch.sum(false_alarm == 0) + torch.sum(mis_detection == 0)) / self.K

        r = torch.zeros((self.K, 1), dtype=h_true.dtype, device=self.device)
        r[active_true, :] = 1
        nmse_numerator = torch.norm(r * h_true.reshape((self.K, -1)) - r * h_est.reshape(self.K, -1), dim=(0, 1)) ** 2
        nmse_denominator = torch.norm(r * h_true.reshape((self.K, -1)), dim=(0, 1)) ** 2
        nmse = nmse_numerator / nmse_denominator
        # nmse = 10 * torch.log10(nmse)

        return nmse, p_error



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=32, help='number of BS antennas')
    parser.add_argument('--N', type=int, default=48, help='number of sub-carriers')
    parser.add_argument('--K', type=int, default=800, help='number of UEs')
    parser.add_argument('--T', type=int, default=30, help='number of time slots')
    parser.add_argument('--snr', type=float, default=10, help='SNR (dB)')
    # parser.add_argument('--P', type=int, default=0.01, help='total power')
    parser.add_argument('--lamda', type=float, default=0.05, help='device active probability')
    parser.add_argument('--delta', type=float, default=0.1, help='noise std. deviation')
    # parser.add_argument('--train_channel_dir', type=str, default='channel_training_dataset_nt32_fft64_real_imag.h5',
    #                     help='directory of training channels')
    parser.add_argument('--test_channel_dir', type=str, default='./jsac_channel_cdl_random_fft48_ula32_test5k.mat',
                         help='directory of testing channels')
    parser.add_argument('--ckpt_path_1st_order', type=str, default='../ncsnv2/exp/logs/quadriga_fft48_ula32/',
                         help='checkpoint path of 1st-order score network')
    parser.add_argument('--ckpt_path_2nd_order', type=str, default='../high_order_dsm/runs/logs/quadriga_fft48_ula32/',
                         help='checkpoint path of 2nd-order score network')
    parser.add_argument('--beta', type=float, default=0.8, help='damping factor')
    parser.add_argument('--ite', type=int, default=30, help='number of iterations')
    parser.add_argument('--ite_gaussian', type=int, default=30, help='number of gaussian iterations')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    args = parser.parse_args()

    args.P = args.delta ** 2 * 10 ** (args.snr / 10)
    np.random.seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # parse config file
    with open(os.path.join('configs', 'unet_channel.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)
    config.device = device
    config.var_h_element = 0.5

    # load first-order score model
    s1 = NCSNv2(config).to(device)
    s1 = torch.nn.DataParallel(s1)
    state_dict_1st_order = torch.load(os.path.join(args.ckpt_path_1st_order, 'checkpoint_15000.pth'),
                                      map_location=device, weights_only=True)
    s1.load_state_dict(state_dict_1st_order[0])
    s1.eval()

    # load second-order score model
    s2 = NCSN(config).to(device)
    s2 = torch.nn.DataParallel(s2)
    state_dict_2nd_order = torch.load(os.path.join(args.ckpt_path_2nd_order, 'checkpoint_550.pth'),
                                      map_location=device, weights_only=True)
    s2.load_state_dict(state_dict_2nd_order[0])
    s2.eval()

    sigmas = get_sigmas_channel(config).to(device)  # different noise std. deviation (real)

    # index of active devices
    r = np.random.binomial(1, args.lamda, size=(args.K, 1, 1))
    active_idx = np.where(r == 1)[0]
    print(len(active_idx))
    print(active_idx)
    active_idx = torch.from_numpy(active_idx).to(device)
    r = torch.from_numpy(r).to(device)

    # channel loading
    h_f, var_h = load_channel(args.test_channel_dir, args.K, args.N, args.M)
    h_f = torch.from_numpy(h_f).to(device)

    # pilot matrix generation
    Q_fft, Q, pilot_sel_idx = pilot_matrix_generation(args.K, args.N, args.T, args.P, device)

    # received signal
    Y = received_signal_generation(h_f, r, Q, args.K, args.N, args.M, args.T, args.delta, device)

    # STMP-JADCE initialization
    stmp_jadce = STMP_JADCE(args, s1, s2, var_h, sigmas, active_idx, device)
    start_time = time.time()
    NMSE, P_error = stmp_jadce.run(Y, Q, h_f, active_idx)
    end_time = time.time()
    print(f"execution time: {(end_time - start_time):.6f} s")

    # Plot P_error values
    plt.plot(range(len(P_error)), P_error, marker='o', color='b', linestyle='-')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Error probability')
    plt.title('Error Plot')

    # Display the plot
    plt.grid(True)
    plt.show()

    # Plot NMSE values
    plt.plot(range(len(NMSE)), 10 * np.log10(NMSE), marker='o', color='r', linestyle='-')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('NMSE')
    plt.title('NMSE Plot')

    # Display the plot
    plt.grid(True)
    plt.show()

    