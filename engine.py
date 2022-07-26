import csv
import gc
import os
import statistics
import time
from pathlib import Path
from tempfile import TemporaryFile


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import mat73
import matplotlib.colors
import pandas as pd
import scipy
from pytorch_lightning.loggers import TensorBoardLogger
from ray.train.trainer import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import HyperBandForBOHB, PopulationBasedTraining, ASHAScheduler
from ray.tune.suggest.bohb import TuneBOHB
from sklearn.linear_model import LinearRegression

from torch.utils.data import TensorDataset, random_split, DataLoader
import hlsvdpropy
import torch
import math
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy.fft as fft
import torch.nn as nn
from scipy.stats import pearsonr, stats
from torchsummary import summary

from utils import Jmrui
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib import rc
from utils import watrem
from Model import Encoder_Model
# %%


class Engine():
    def __init__(self, parameters):
        if parameters['intr_plot'] == False:
            plt.ioff()
        else:
            plt.ion()
        self.parameters = parameters
        self.saving_dir = parameters['parent_root'] + parameters['child_root'] + parameters['version']
        self.epoch_dir =  "epoch/"
        self.loging_dir = parameters['parent_root'] + parameters['child_root']
        self.data_dir = parameters['data_dir']
        self.data_dir_ny = parameters['data_dir_ny']
        self.basis_dir = parameters['basis_dir']
        self.test_data_root = parameters['test_data_root']
        Path(self.saving_dir).mkdir(parents=True, exist_ok=True)
        Path(self.saving_dir+self.epoch_dir).mkdir(parents=True, exist_ok=True)
        self.max_epoch = parameters['max_epoch']
        self.batchsize = parameters['batchsize']
        self.numOfSample = parameters['numOfSample'];
        self.t_step = parameters['t_step']
        self.trnfreq = parameters['trnfreq']
        self.nauis = parameters['nauis']
        self.save = parameters['save']
        self.tr = parameters['tr']
        self.betas = parameters['betas']
        self.depths = parameters['depths']
        self.ens = parameters['ens']
        self.met_name = parameters['met_name']
        self.BW = 1 / self.t_step
        self.f = np.linspace(-self.BW / 2, self.BW / 2, 4096)
        try:
            basis_name = parameters["basis_name"]
        except:
            basis_name = "data"
        if self.basis_dir is not None:
            self.basisset = (sio.loadmat(self.basis_dir).get(basis_name))
            if parameters['basis_conj']:
                self.basisset = np.conj(self.basisset)
            try:
                if parameters['norm_basis']:
                    pass
                    # self.basisset = self.normalize(self.basisset)
            except:
                pass
        self.wr = parameters['wr']
        self.data_name = parameters['data_name']
        if self.data_dir is not None:
            try:
                self.dataset = scipy.io.loadmat(self.data_dir).get(self.data_name)
            except:
                self.dataset = mat73.loadmat(self.data_dir).get(self.data_name)
        self.numOfSig = parameters['numOfSig']
        self.sigLen = parameters['sigLen']
        self.truncSigLen = parameters['truncSigLen']
        self.BW = 1 / self.t_step
        self.f = np.linspace(-self.BW / 2, self.BW / 2, self.sigLen)
        self.t = np.arange(0, self.sigLen) * self.t_step
        self.t = np.expand_dims(self.t, 1)
        self.MM = parameters['MM']
        self.MM_f = parameters['MM_f']
        self.MM_d = np.array(parameters['MM_d'])
        self.MM_a = parameters['MM_a']
        self.MM_plot = parameters['MM_plot']
        self.pre_plot = parameters['pre_plot']
        self.basis_need_shift = parameters['basis_need_shift']
        self.aug_params = parameters['aug_params']
        self.tr_prc = parameters['tr_prc']
        self.in_shape= parameters['in_shape']
        self.enc_type = parameters['enc_type']
        self.banorm = parameters['banorm']
        if self.basis_dir and parameters['max_c'] is not None:
            max_c = np.array(parameters['max_c'])
            min_c = np.array(parameters['min_c'])
            self.min_c = (min_c) / np.max((max_c));
            self.max_c = (max_c) / np.max((max_c));
        self.reg_wei = parameters['reg_wei']
        self.data_conj = parameters['data_conj']
        self.test_nos = parameters['test_nos']
        self.quality_filt = parameters['quality_filt']
        self.test_name = parameters['test_name']
        self.beta_step = parameters['beta_step']
        self.MM_type = parameters['MM_type']
        self.MM_dir = parameters['MM_dir']
        self.MM_constr = parameters['MM_constr']
        self.comp_freq = parameters['comp_freq']
        if self.MM_dir is not None:
            self.mm = sio.loadmat(self.MM_dir).get("data")
            self.mm[0] = self.mm[0] - 1*fft.fftshift(fft.fft(self.mm, axis=0))[0]
        self.sim_params = parameters['sim_params']
        if self.sim_params is not None:
            for i, val in enumerate(self.sim_params):
                if isinstance(val,str):
                    self.sim_params[i] = getattr(self, self.sim_params[i])
        try:
            self.test_params = parameters['test_params']

            if self.test_params is not None:
                for i, val in enumerate(self.test_params):
                    if isinstance(val,str):
                        self.test_params[i] = getattr(self, self.test_params[i])
        except:
            pass

        if self.MM:
            if parameters['MM_model'] == "lorntz":
                self.MM_model = self.Lornz
                self.MM_d = (np.pi * self.MM_d)
                # self.MM_d = (np.pi ** self.MM_d) * ((self.MM_d) ** 2) / (2 * np.log(2))
            if parameters['MM_model'] == "gauss":
                self.MM_model = self.Gauss
                self.MM_d = self.MM_d * self.trnfreq
                self.MM_d = (np.pi * self.MM_d)/(2*np.sqrt(np.log(2)))
            self.numOfMM = len(self.MM_f)
            if self.MM_type == 'single' or self.MM_type == 'single_param':
                self.met_name.append("MM")
        self.heatmap_cmap = sns.diverging_palette(20, 220, n=200)
        self.sim_now = parameters['sim_order'][0]
        self.sim_dir = parameters['sim_order'][1]
        try:
            self.kw = self.parameters['kw']
        except:
            self.parameters['kw'] = 5
            self.kw = 5
        try:
            self.MM_fd_constr = self.parameters['MM_fd_constr']
        except:
            self.MM_fd_constr = True

        try:
            self.MM_conj = self.parameters['MM_conj']
        except:
            self.MM_conj = True
            print("take care MM is conjucated!")

        if self.MM_conj == False:
            self.MM_f = [zz - 4.7 for zz in  self.MM_f]

        if self.basis_need_shift[0] == True:
            self.basisset = self.basisset[0:2048, :] * np.exp(
                2 * np.pi * self.ppm2f(self.basis_need_shift[1]) * 1j * self.t)

            # %%
    def getSignals(self,min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond):
        """
        It takes in a bunch of parameters and returns a bunch of signals

        :param min_c: minimum amplitude of the signal
        :param max_c: maximum amplitude of the signal
        :param f: frequency shift
        :param d: the decay rate of the signal
        :param ph: phase shift
        :param noiseLevel: the standard deviation of the noise
        :param ns: number of signals
        :param mm_cond: if True, then the signal will include a macromolecule
        :return: the signal, the amplitudes, the shifts, the alphas, and the phases.
        """
        if mm_cond == True:
            basisset = np.concatenate((self.basisset, self.mm), axis=1)
            numOfSig = self.numOfSig + 1
        ampl = min_c + np.multiply(np.random.random(size=(ns, numOfSig)), (max_c - min_c))
        # ampl = np.multiply(np.random.random(size=(ns, numOfSig)), (max_c))
        shift = f * np.random.rand(ns) - f / 2
        freq = -2 * math.pi * (shift) * self.t
        alpha = d * np.random.rand(ns)
        ph = (ph * np.random.rand(ns) * math.pi) - (ph / 2 * math.pi)
        signal = np.matmul(ampl[:, 0:numOfSig], basisset[0:self.sigLen, :].T)
        noise = np.random.normal(0, noiseLevel, (ns, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (ns, self.sigLen))
        signal = signal + noise

        y = np.multiply(signal, np.exp(freq * 1j).T)
        y = y.T * np.exp(ph * 1j)
        y = np.multiply(y, np.exp(-alpha * self.t))
        return y, ampl, shift, alpha, ph

    def getSignal(self,ampl, shift, alpha, ph, noiseLevel,mm_cond):
        """
        The function takes in a set of parameters and returns a signal with noise

        :param ampl: Amplitude of the signal
        :param shift: the frequency shift of the signal
        :param alpha: the decay rate of the signal
        :param ph: phase of the signal
        :param noiseLevel: the standard deviation of the noise
        :param mm_cond: if True, then the signal will include a macromolecule
        :return: the signal with noise and the signal without noise.
        """
        if mm_cond == True:
            basisset = np.concatenate((self.basisset, self.mm), axis=1)
            numOfSig = self.numOfSig + 1
        freq = -2 * math.pi * (shift) * self.t
        signal = np.matmul(ampl[:, 0:numOfSig], basisset[0:self.sigLen, :].T)
        noise = np.random.normal(0, noiseLevel, (1, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (1, self.sigLen))
        y = np.multiply(signal, np.exp(freq * 1j).T)
        y = y.T * np.exp(ph * 1j)
        y = np.multiply(y, np.exp(-alpha * self.t))
        return y + noise.T, y

    def get_augment(self, signals, n, f_band, ph_band, ampl_band, d_band, noiseLevel):
        """
        It takes a signal, and creates a number of augmented signals by adding noise, shifting the frequency, and shifting
        the phase

        :param signals: the original signal
        :param n: number of samples to generate for each signal
        :param f_band: frequency band
        :param ph_band: phase shift
        :param ampl_band: the amplitude of the signal can vary by this amount
        :param d_band: the maximum value of the exponential decay parameter
        :param noiseLevel: the standard deviation of the noise
        :return: the augmented signal, the amplitude, the delay, the shift and the phase.
        """
        l = []
        l.append(signals)
        lens = np.shape(signals)[1]
        shift_t = f_band * np.random.rand(n * lens) - (f_band / 2)
        ph_t = ph_band * np.random.rand(n * lens) * math.pi - ((ph_band / 2) * math.pi)
        ampl_t = 1 + ((ampl_band * np.random.rand(n * lens))-ampl_band/2)
        d_t = d_band * np.random.rand(n * lens)
        for i in range(0, lens):
            signal = np.expand_dims(signals[:, i], 1)
            numOfSamplei = n
            freq = -2 * math.pi * (shift_t[i * numOfSamplei:(i + 1) * numOfSamplei]) * self.t
            ph = ph_t[i * numOfSamplei:(i + 1) * numOfSamplei]
            ampl = ampl_t[i * numOfSamplei:(i + 1) * numOfSamplei]
            d = d_t[i * numOfSamplei:(i + 1) * numOfSamplei]
            y = ampl * signal
            y = np.multiply(y * np.exp(ph * 1j), np.exp(freq * 1j))
            y = np.multiply(y, np.exp(-d * self.t))
            noise = np.random.normal(0, noiseLevel, (len(signal), numOfSamplei)) + 1j * np.random.normal(0, noiseLevel,
                                                                                                         (len(signal),
                                                                                                          numOfSamplei))
            y_i = y + noise
            l.append(y_i)
        y = np.hstack(l)
        return y, ampl_t, d_t, shift_t, ph_t

    def savefig(self, path, plt_tight=True):
        """
        It saves the current figure to the specified path, with the specified format, and with the specified dpi

        :param path: the name of the file to save
        :param plt_tight: If True, it will make the plot look nice, defaults to True (optional)
        """
        # plt.ioff()
        if plt_tight:
            plt.tight_layout()
        if self.save:
            plt.savefig(self.saving_dir + path + ".svg", format="svg")
            plt.savefig(self.saving_dir + path + " .png", format="png", dpi=1200)
        plt.clf()
        # plt.show()


    # %%
    def loadModel(autoencoder, path):
        """
        > Loads a model from a file

        :param autoencoder: the model to be loaded
        :param path: the path to the model file
        :return: The model is being returned.
        """
        # m = LitAutoEncoder(t,signal_norm)
        return autoencoder.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # %%
    def tic(self):
        global start_time
        start_time = time.time()

    def toc(self,name):
        elapsed_time = (time.time() - start_time)
        print("--- %s seconds ---" % elapsed_time)
        timingtxt = open(self.saving_dir + name + ".txt", 'w')
        timingtxt.write(name)
        timingtxt.write("--- %s ----" % elapsed_time)
        timingtxt.close()

    # %%
    def cal_snr(self,data, endpoints=128,offset=0):
        """
        It takes the first point of the data and divides it by the standard deviation of the last 128 points of the data

        :param data: the data you want to calculate the SNR for
        :param endpoints: The number of points at the beginning and end of the data to use for calculating the noise,
        defaults to 128 (optional)
        :param offset: The number of points to skip at the beginning of the data, defaults to 0 (optional)
        :return: The signal to noise ratio of the data.
        """
        return np.abs(data[0, :]) / np.std(data.real[-(offset + endpoints):-(offset+1), :], axis=0)

    def cal_snrf(self,data_f,endpoints=128,offset=0):
        """
        It takes the absolute value of the maximum value of the data, and divides it by the standard deviation of the data

        :param data_f: the data in the frequency domain
        :param endpoints: the number of points to use for the standard deviation calculation, defaults to 128 (optional)
        :param offset: The offset of the data to be used for calculating the SNR, defaults to 0 (optional)
        :return: The max of the absolute value of the data_f divided by the standard deviation of the real part of the
        data_f.
        """
        return np.max(np.abs(data_f), 0) / (np.std(data_f.real[offset:endpoints+offset, :],axis=0))

    def ppm2p(self, r, len):
        r = 4.7 - r
        return int(((self.trnfreq * r) / (1 / (self.t_step * len))) + len / 2)

    def ppm2f(self, r):
        return r * self.trnfreq

    def fillppm(self, y1, y2, ppm1, ppm2, rev, alpha=.1, color='red'):
        p1 = int(self.ppm2p(ppm1, len(y1)))
        p2 = int(self.ppm2p(ppm2, len(y1)))
        n = p2 - p1
        x = np.linspace(int(ppm1), int(ppm2), abs(n))
        plt.fill_between(np.flip(x), y1[p2:p1, 0].real,
                         y2[p2:p1, 0].real, alpha=0.1, color='red')
        if rev:
            plt.gca().invert_xaxis()

    def plotsppm(self, sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-', color=None):
        p1 = int(self.ppm2p(ppm1, len(sig)))
        p2 = int(self.ppm2p(ppm2, len(sig)))
        n = p2 - p1
        x = np.linspace(int(ppm1), int(ppm2), abs(n))
        sig = np.squeeze(sig)
        df = pd.DataFrame(sig[p2:p1, :].real)
        df['Frequency(ppm)'] = np.flip(x)
        df_m = df.melt('Frequency(ppm)',value_name='Real Signal (a.u.)')
        g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df_m, linewidth=linewidth, linestyle=linestyle)
        plt.legend([],[],frameon=False)
        # g = plt.plot(np.flip(x), sig[p2:p1, :].real, linewidth=linewidth, linestyle=linestyle, color=color)
        if rev:
            plt.gca().invert_xaxis()

    def normalize(self,inp):
        """
        It takes the absolute value of the input, divides it by the maximum absolute value of the input, and then multiplies
        it by the exponential of the angle of the input

        :param inp: The input signal
        :return: The magnitude of the input divided by the maximum magnitude of the input, multiplied by the complex
        exponential of the angle of the input.
        """
        return (np.abs(inp) / np.abs(inp).max(axis=0)) * np.exp(np.angle(inp) * 1j)

    def plotppm(self, sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-'):
        p1 = int(self.ppm2p(ppm1, len(sig)))
        p2 = int(self.ppm2p(ppm2, len(sig)))
        n = p2 - p1
        x = np.linspace(int(ppm1), int(ppm2), abs(n))
        sig = np.squeeze(sig)
        df = pd.DataFrame({'Real Signal (a.u.)': sig[p2:p1].real})
        df['Frequency(ppm)'] = np.flip(x)
        g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df, linewidth=linewidth, linestyle=linestyle)
        if rev:
            plt.gca().invert_xaxis()
        return g
        # gca = plt.plot(x,sig[p2:p1,0],linewidth=linewidth, linestyle=linestyle)

    def plot_basis2(self, basisset, ampl):
        basisset = self.normalize(basisset)
        for i in range(0, len(basisset.T) - 1):
            self.plotppm(-1* i + fft.fftshift(fft.fft(ampl * basisset[:, i])), 0, 10, False)
        self.plotppm(-20 * (i + 1) + fft.fftshift(fft.fft(basisset[:, i + 1])), 0, 10, True)
        plt.legend(self.met_name)
        self.savefig("Basis" + str(ampl))
        

    def plot_basis(self, ampl, fr, damp, ph, ):
        for i in range(0, len(self.basisset.T)):
            vv=fft.fftshift(fft.fft(ampl[0, i] * self.basisset[:, i]*np.exp(-2 * np.pi *1j* fr * self.t.T)*np.exp(-1*damp*self.t.T)))
            ax = self.plotppm(-2.2 * (i+2) + vv.T, 0, 5, False)
            sns.despine(left=True,right=True,top=True)
            plt.text(.1, -2.2 * (i+2), self.met_name[i],fontsize=8)
            ax.tick_params(left=False)
            ax.set(yticklabels=[])

    def Lornz(self, ampl, f, d, ph, Crfr, Crd):
        return ampl * np.multiply(np.multiply(np.exp(ph * 1j),
                                                    np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                     np.exp(-1*(d + Crd) * self.t.T))
    def Gauss(self, ampl, f, d, ph, Crfr, Crd):
        return ampl * np.multiply(np.multiply(np.exp(ph * 1j),
                                                    np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                     np.exp(-1*((d + Crd)**2) * self.t.T * self.t.T))
    def data_proc(self):
        """
        It takes in a dataset, and returns a training set and a test set.

        The training set is the first part of the dataset, and the test set is the last part of the dataset.

        The test set is also saved as a .mat file, and a .jmrui file.

        :return: y is the training data, y_test is the test data
        """
        if self.wr[0] == True:
            self.dataset = watrem.init(self.dataset[:,:],self.t_step, self.wr[1])
            with open(self.data_dir_ny, 'wb') as f:
                np.save(f, self.dataset, allow_pickle=True)
        else:
            if self.data_dir_ny is not None:
                with open(self.data_dir_ny, 'rb') as f:
                    self.dataset = np.load(f)
        if self.data_conj == True:
            y = np.conj(self.dataset)
        else:
            y = self.dataset
        # y = y[4:,:]
        y = self.normalize(y)
        ang = np.angle(y[1, :])
        y = y * np.exp(-1 * ang * 1j)

        y_test =  y[:,- self.test_nos:]
        y = y[:, 0:-self.test_nos]

        y_f = fft.fftshift(fft.fft(y, axis=0), axes=0)

        if self.quality_filt[0] == True:
            cond = np.mean(np.abs(y_f[self.quality_filt[1]:self.quality_filt[2], :]), axis=0)
            # con2 = np.mean(np.abs(y_f[self.quality_filt[3]:self.quality_filt[4], :]), axis=0)
            idx = np.where((cond < 6))[0]
            y = y[0:2 * self.truncSigLen, idx]



        snrs = self.cal_snrf(fft.fftshift(fft.fft(y_test, axis=0), axes=0))
        y_f = fft.fftshift(fft.fft(y_test, axis=0), axes=0)
        cond = np.mean(np.abs(y_f[self.quality_filt[1]:self.quality_filt[2], :]), axis=0)
        self.y_test_idx = np.where((cond < 6))[0]

        data = []
        data.append(y_test)
        data.append(snrs)
        data.append(self.y_test_idx)

        np.savez(self.saving_dir + "test_" + str(self.test_nos), *data)

        sio.savemat(self.saving_dir + "test_" + str(self.test_nos) + "_testDB.mat",
                    {'y_test': y_test, 'snrs_t': snrs})

        Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1),
                                     self.t_step * 1000, 0, 0, self.trnfreq*1e6), y_test , self.saving_dir + self.test_name)

        Jmrui.write(Jmrui.makeHeader("basis_set", np.size(self.basisset, 0), np.size(self.basisset, 1),
                                     self.t_step * 1000, 0, 0, self.trnfreq*1e6), self.basisset , self.saving_dir + "basis_set")
        np.save(self.saving_dir + "basis_set", self.basisset)

        return y,y_test
    def data_prep(self):
        """
        This function is used to prepare the data for the model.

        The first thing it does is check if the simulation parameters are not None. If they are not None, then it checks if
        the simulation is now. If the simulation is now, then it gets the signals using the getSignals function. It then
        saves the data in a .npz file. If the simulation is not now, then it tries to load the data from the .npz file. If
        it can't find the .npz file, then it prints an error message.

        If the simulation parameters are None, then it gets the data using the data_proc function.

        If the augmentation parameters are not None, then it augments the data using the data_aug function.

        It then takes the fft of the data.

        If the pre_plot parameter is True, then it plots the histogram of the signal to noise
        """
        if self.sim_params is not None:
            if self.sim_now == True:
                data = self.getSignals(self.sim_params[0],self.sim_params[1],self.sim_params[2]
                                                ,self.sim_params[3],self.sim_params[4],self.sim_params[5],
                                                self.sim_params[6], True)
                np.savez(self.sim_dir+str(self.sim_params[2:]),*data)
                y, self.ampl_t, shift, self.alpha, ph = data
            else:
                try:
                    data = np.load(self.sim_dir+str(self.sim_params[2:])+'.npz')
                    y, self.ampl_t, shift, self.alpha, ph = [data[x] for x in data]
                except:
                    print('there is no data in the sim dir')

            # y = self.normalize(y)
        else:
            y_train, self.y_test = self.data_proc()
        if self.aug_params is not None:
            y, _, _, _, _ = self.data_aug(y_train[0:self.sigLen,:])
        y_f = fft.fftshift(fft.fft(y, axis=0),axes=0)
        if self.pre_plot ==True:
            plt.hist(self.cal_snrf(y_f))
            plt.show()
            plt.close()
            self.plotppm(fft.fftshift(fft.fft((y[:, 0]), axis=0)), 0, 5, True, linewidth=1, linestyle='-')
            plt.plot(range(0, 512), y[0:512, 0])
            
        self.numOfSample = np.shape(y)[1];
        y_norm = y
        del y
        self.to_tensor(y_norm)
    def data_aug(self,y):
        return self.get_augment(y, self.aug_params[0], self.aug_params[1], self.aug_params[2], self.aug_params[3], self.aug_params[4], self.aug_params[5])
    def to_tensor(self,y_norm):
        """
        It takes the normalized signal and truncates it to the length of the truncated signal.

        It then converts the truncated signal to a tensor.

        If the simulation parameters are not None, it splits the truncated signal into a training and validation set.

        If the simulation parameters are None, it splits the truncated signal into a training and validation set.

        :param y_norm: the normalized signal
        """
        y_trun = y_norm[0:self.truncSigLen, :].astype('complex64')
        self.y_trun = torch.from_numpy(y_trun[:, 0:self.numOfSample].T)


        if self.sim_params is None:
            y_test = self.y_test[0:self.truncSigLen, :].astype('complex64')
            self.y_test_trun = torch.from_numpy(y_test[:, 0:self.numOfSample].T)
            self.train = TensorDataset(self.y_trun)
            self.val = TensorDataset(self.y_test_trun)
        else:
            self.y_test_trun = self.y_trun
            my_dataset = TensorDataset(self.y_trun,torch.from_numpy(np.hstack((self.ampl_t,np.expand_dims(self.alpha,1)))))
            self.train, self.val = random_split(my_dataset, [int((self.numOfSample) * self.tr_prc), self.numOfSample - int((self.numOfSample) * self.tr_prc)])


    def inputSig(self,x):
        if self.in_shape == '2chan':
            return torch.cat((torch.unsqueeze(x[:, 0:self.truncSigLen].real, 1), torch.unsqueeze(x[:, 0:self.truncSigLen].imag, 1)),1)
        if self.in_shape == 'real':
            return x[:, 0:self.truncSigLen].real
    def calib_plot(self,ampl_t,y_out, yerr,cmap):
        if cmap==None :
            ax = plt.scatter(x=ampl_t, y=y_out)
        else:
            ax = plt.scatter(x=ampl_t, y=y_out, c=yerr, cmap='Spectral')
            plt.set_cmap(cmap)
            cb = plt.colorbar()
            cb.outline.set_visible(False)
        ax.axes.spines['right'].set_visible(False)
        ax.axes.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.axes.yaxis.set_ticks_position('left')
        ax.axes.xaxis.set_ticks_position('bottom')
        x0, x1 = ax.axes.get_xlim()
        y0, y1 = ax.axes.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.axes.plot(lims, lims, '--k')
        ax.axes.set_xlabel("True")
        ax.axes.set_ylabel('Predicted')
    def err_plot(self, x , y, yerr, name, cmap):
        if cmap==None :
            ax = plt.scatter(x, y)
        else:
            ax = plt.scatter(x, y, c=yerr, cmap='Spectral')
            cb = plt.colorbar()
            cb.outline.set_visible(False)
        plt.set_cmap(cmap)
        ax.axes.spines['right'].set_visible(False)
        ax.axes.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.axes.yaxis.set_ticks_position('left')
        ax.axes.xaxis.set_ticks_position('bottom')
        ax.axes.set_xlabel(name)
        ax.axes.set_ylabel('Prediction Error')
    def test_compact(self):
        """
        The function takes in a set of parameters and generates a test set of signals.

        The parameters are:

        - min_c: minimum concentration of metabolites
        - max_c: maximum concentration of metabolites
        - f: frequency of the signal
        - d: damping factor of the signal
        - ph: phase of the signal
        - noiseLevel: noise level of the signal
        - ns: number of signals
        - mm_cond: whether to include MM or not

        The function then generates a test set of signals and saves it in a .npz file.

        The function then loads the test set of signals and passes it to the predict_ensembles function.

        The predict_ensembles function returns the predicted values of the metabolites, the frequencies, damping factors,
        phases, and the encodings and decodings of the signals.

        The function then saves the predicted values of the
        """
        cmap = 'Blues'
        id = "test_" + str(self.test_params[6]) + "_" + str(self.test_params[5]) + "_"
        if self.parameters['test_load']:
            data = np.load(self.sim_dir+"test_"+str(self.test_params[2:])+'.npz')
            y_test, ampl_t, shift_t, alpha_t, ph_t, snrs = [data[x] for x in data]
        else:
            # min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond
            data = self.getSignals(self.test_params[0], self.test_params[1], self.test_params[2]
                       ,self.test_params[3], self.test_params[4], self.test_params[5],
                       self.test_params[6], True)
            y_test, ampl_t, shift_t, alpha_t, ph_t = data
            snrs = self.cal_snrf(fft.fftshift(fft.fft(y_test, axis=0),axes=0))
            data=list(data)
            data.append(snrs)
            np.savez(self.sim_dir+"test_"+str(self.test_params[2:]),*data)
            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), y_test,
                        self.sim_dir+"test_"+str(self.test_params[2:]) + '_testDBjmrui.txt')
            sio.savemat(self.sim_dir+"test_"+str(self.test_params[2:]) + "_testDB.mat",
                        {'y_test': y_test, 'ampl_t': ampl_t, 'shift_t': shift_t, 'alpha_t': alpha_t, 'ph_t': ph_t,'snrs_t': snrs, })

        id = "test/" + id + "/"
        selected_met = ["Cr", "GPC", "Glu", "Ins", "NAA", "NAAG", "PCho", "PCr", "Tau"]
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        y_test = y_test.astype('complex64')
        y_test = torch.from_numpy(y_test)
        # print(y_test.size())
        self.tic()
        rslt = self.predict_ensembles(y_test)
        self.toc(id + "time")
        y_out,  fr, damp, ph, decs, encs = rslt

        # test_info[self.met_name] = np.abs((ampl_t - y_out)/(ampl_t + y_out))*100
        if self.MM == True:
            self.numOfComp = self.numOfSig +1

        test_info = pd.DataFrame()
        test_info['SNR'] = snrs
        test_info['Frequency'] = shift_t
        test_info['Damping'] = alpha_t
        test_info['Phase'] = ph_t
        test_info[self.met_name] = ampl_t[:, 0:len(self.met_name)]
        test_info['type'] = 'True'
        test_temp = pd.DataFrame()
        test_temp['SNR'] = snrs
        test_temp['Frequency'] = fr
        test_temp['Damping'] = damp
        test_temp['Phase'] = ph
        test_temp[self.met_name] = y_out
        test_temp['type'] = 'Predicted'
        test_info = test_info.append(test_temp)
        test_info.to_csv(self.saving_dir + id + "rslt.csv")
        np.savez(self.saving_dir + id + "rslt.npz",*rslt)
        errors_DL = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2','intercept','coef'], index=self.met_name)
        for i in range(0, self.ens):
            for j in range(0, self.numOfSig):
                # ax = sns.regplot(x=ampl_t[:, j], y=encs[i, :, j],label=str(i))
                model = LinearRegression().fit(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1)))
                errors_DL.iloc[j] = [r2_score(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_squared_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_percentage_error(ampl_t[:, j], encs[i, :, j]) * 100,
                                     model.score(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1))),
                                     model.intercept_[0],
                                     model.coef_[0][0]]
            errors_DL.to_csv(self.saving_dir + id + "_" + str(i) + "Ens_errorsDL.csv")






    def test(self):
        """
        The function takes in the test data and the parameters of the model and saves the results of the test in a folder
        """
        cmap = 'Blues'
        id = "test_" + str(self.test_params[6]) + "_" + str(self.test_params[5]) + "_"
        data = np.load(self.sim_dir + "test_" + str(self.test_params[2:]) + '.npz')
        y_test, ampl_t, shift_t, alpha_t, ph_t, snrs = [data[x] for x in data]

        id = "test_all/" + id + "/"
        selected_met = ["Cr", "GPC", "Glu", "Ins", "NAA", "NAAG", "PCho", "PCr", "Tau"]
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        test_info = pd.DataFrame()
        test_info['Frequency'] = np.abs(shift_t)
        test_info['Damping'] = alpha_t
        test_info['Phase'] = np.abs(ph_t)
        test_info['SNR'] = snrs
        sns.set(style="white", palette="muted", color_codes=True)
        sns.distplot(test_info['SNR'], color="m")
        sns.despine()
        self.savefig("test_snr_hist")
        
        y_test = y_test.astype('complex64')
        y_test = torch.from_numpy(y_test)
        # print(y_test.size())
        self.tic()
        y_out, fr, damp, ph, decs, encs = self.predict_ensembles(y_test)
        self.toc(id + "time")
        # test_info[self.met_name] = np.abs((ampl_t - y_out)/(ampl_t + y_out))*100
        if self.MM == True:
            self.numOfComp = self.numOfSig +1
        test_info[self.met_name] = np.abs(ampl_t[:,0:len(self.met_name)] - y_out)
        type = ['Predicted' for i in y_out]
        net_pred = pd.DataFrame(y_out,columns=self.met_name)
        net_pred['type'] = type
        type = ['True' for i in y_out]
        net_true = pd.DataFrame(ampl_t[:,0:len(self.met_name)],columns=self.met_name)
        net_true['type'] = type
        net_pred = net_pred.append(net_true)
        dfm = pd.melt(net_pred, id_vars=['type'])

        lc = [self.met_name[i] for i in (np.where(self.max_c[0:len(self.met_name)] < 0.3)[0])]
        sns.set_style('whitegrid')
        sns.violinplot(x='variable', y='value', data=dfm[dfm['variable'].isin(lc)], hue='type', palette="Set3",
                       linewidth=1,
                       split=True,
                       inner="quartile")
        sns.despine()
        self.savefig(id + "violion_low")
        
        lc = [self.met_name[i] for i in (np.where((self.max_c[0:len(self.met_name)]  > 0.3) & (self.max_c[0:len(self.met_name)]  < 1.01))[0])]
        sns.violinplot(x='variable', y='value', data=dfm[dfm['variable'].isin(lc)], hue='type', palette="Set3",
                       linewidth=1,
                       split=True,
                       inner="quartile")
        sns.despine()
        self.savefig(id + "violion_high")
        
        corr = test_info.corr()
        corr.iloc[4:, 0:4].transpose().to_csv(self.saving_dir + id + "_errors_corr.csv")
        sns.heatmap(data=corr.iloc[4:, 0:4].transpose(), cmap=self.heatmap_cmap)
        self.savefig(id + "corrollation_heatmap")
        

        # quest, true = rslt_vis.getQuest()
        errors_DL = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2','intercept','coef'], index=self.met_name)
        errors_Q = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2','intercept','coef'], index=self.met_name)
        for i in range(0, self.ens):
            for j in range(0, self.numOfSig):
                # ax = sns.regplot(x=ampl_t[:, j], y=encs[i, :, j],label=str(i))
                model = LinearRegression().fit(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1)))
                errors_DL.iloc[j] = [r2_score(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_squared_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_percentage_error(ampl_t[:, j], encs[i, :, j]) * 100,
                                     model.score(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1))),
                                     model.intercept_[0],
                                     model.coef_[0][0]]
                # plt.title("DL::" + str(i) + "::"+ self.met_name[j])
                # # plt.legend()
                # x0, x1 = ax.get_xlim()
                # y0, y1 = ax.get_ylim()
                # lims = [max(x0, y0), min(x1, y1)]
                # ax.plot(lims, lims, '--k')
                # plt.show()
            errors_DL.to_csv(self.saving_dir + id + "_" + str(i) + "Ens_errorsDL.csv")

        file = open(self.saving_dir + id + '_predicts.csv', 'w')
        writer = csv.writer(file)
        writer.writerows(np.concatenate((y_out, fr, damp, ph), axis=1))
        file.close()
        mean_f = np.mean((fr) - np.expand_dims(shift_t, axis=[1]))
        mean_alph = np.mean((damp) - np.expand_dims(alpha_t, axis=[1]))
        mean_ph = np.mean((ph) - np.expand_dims(ph_t, axis=[1]))
        std_f = np.std((fr) - np.expand_dims(shift_t, axis=[1]))
        std_alph = np.std((damp) - np.expand_dims(alpha_t, axis=[1]))
        std_ph = np.std((ph) - np.expand_dims(ph_t, axis=[1]))



        file = open(self.saving_dir + id + '_rslt.csv', 'w')
        writer = csv.writer(file)
        writer.writerow(["freq", mean_f, std_f])
        writer.writerow(["damp", mean_alph, std_alph])
        writer.writerow(["ph", mean_ph, std_ph])
        # test_info['f_error'] = fr[:, 0] - np.expand_dims(shift_t, axis=[1])[:, 0]
        # test_info['d_error'] = damp[:, 0] - np.expand_dims(alpha_t, axis=[1])[:, 0]
        # test_info['p_error'] = (ph[:, 0] * 180 / np.pi) - np.expand_dims(ph_t * 180 / np.pi, axis=[1])[:, 0]
        # sns.jointplot(x=test_info[['f_error','d_error']], y=test_info[['f','d']])
        ax = sns.jointplot(x=np.expand_dims(shift_t, axis=[1])[:, 0], y=fr[:, 0] - np.expand_dims(shift_t, axis=[1])[:, 0])
        ax.ax_joint.set_xlabel('Frequency')
        ax.ax_joint.set_ylabel('Prediction Error')
        self.savefig(id + "freq")
        
        ax = sns.jointplot(x=np.expand_dims(alpha_t, axis=[1])[:, 0], y=damp[:, 0] - np.expand_dims(alpha_t, axis=[1])[:, 0])

        ax.ax_joint.set_xlabel('Damping')
        ax.ax_joint.set_ylabel('Prediction Error')
        self.savefig(id + "damp")
        
        ax = sns.jointplot(x=np.expand_dims(ph_t * 180 / np.pi, axis=[1])[:, 0],
                      y=(ph[:, 0] * 180 / np.pi) - np.expand_dims(ph_t * 180 / np.pi, axis=[1])[:, 0])
        ax.ax_joint.set_xlabel('Phase')
        ax.ax_joint.set_ylabel('Prediction Error')
        self.savefig(id + "ph")
        


        ids1 = [2, 12, 8, 14, 17, 9]
        ids2 = [15, 13, 7, 5, 6, 10]
        names = ["Cr+PCr", "NAA+NAAG", "Glu+Gln", "PCho+GPC", "Glc+Tau", "Ins+Gly"]
        errors_combined = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2','intercept','coef'], index=names)
        idx = 0
        for id1, id2, name in zip(ids1, ids2, names):
            # var = (y_out_var[:, id1]**2 + y_out_var[:, id2]**2) + (ampl_t[:, id1]**2 + ampl_t[:, id2]**2)
            # corr, _ = pearsonr(ampl_t[:, id1], ampl_t[:, id2])
            # warning! how we can calculate sd for two corrolated normal distribution!?
            # sd = 100 * np.sqrt(y_out_var[:, id1] + y_out_var[:, id2]) / (y_out[:, id1] + y_out[:, id2])
            self.calib_plot(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2]), None, None)

            plt.title(name)
            self.savefig(id + "combined_" + name)
            
            model = LinearRegression().fit((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)), (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1)))
            errors_combined.iloc[idx] = [r2_score(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2])),
                                         mean_absolute_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                             (y_out[:, id1] + y_out[:, id2])),
                                         mean_squared_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                            (y_out[:, id1] + y_out[:, id2])),
                                         mean_absolute_percentage_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                                        (y_out[:, id1] + y_out[:, id2])) * 100,
                                         model.score((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)), (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1))),
                                         model.intercept_,
                                         model.coef_
                                         ]

            idx += 1
        errors_combined.to_csv(self.saving_dir + id + "_errors_combined.csv")
        errors_averaged = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2','intercept','coef'], index=self.met_name)
        if self.parameters["detailed_test"]:
            j = 0
            errors_corr = pd.DataFrame(columns=self.met_name, index=["damping", 'frequency', 'Phase', 'SNR'])
            for idx, name in enumerate(self.met_name):
                model = LinearRegression().fit(ampl_t[:, j].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1)))
                errors_averaged.iloc[j] = [r2_score(ampl_t[:, idx], y_out[:, idx]),
                                           mean_absolute_error(ampl_t[:, idx], y_out[:, idx]),
                                           mean_squared_error(ampl_t[:, idx], y_out[:, idx]),
                                           mean_absolute_percentage_error(ampl_t[:, idx], y_out[:, idx]) * 100,
                                           model.score(ampl_t[:, j].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1))),
                                           model.intercept_,
                                           model.coef_
                                           ]

                self.calib_plot(ampl_t[:, idx], y_out[:, idx], None, None)
                plt.title(self.met_name[idx])
                self.savefig(id + "seperated_percent" + name)


                self.calib_plot(ampl_t[:, idx], y_out[:, idx], None, None)
                plt.title(self.met_name[idx])
                self.savefig(id + "seperated" + name)

                
                j += 1

            for idx_null,name in enumerate(selected_met):
                idx = self.met_name.index(name)
                err = np.abs(y_out[:, idx] - ampl_t[:, idx])


                self.err_plot(np.expand_dims(alpha_t, axis=[1])[:, 0], err, None, 'Damping',None)
                plt.title(name)
                self.savefig(id + "corrollation_precent" + 'Damping_' + name)
                

                self.err_plot(np.expand_dims(shift_t, axis=[1])[:, 0], err, None, 'Frequency',None)
                plt.title(name)
                self.savefig(id + "corrollation_precent" + 'Shift_' + name)
                

                self.err_plot(np.expand_dims(ph_t*(180/np.pi), axis=[1])[:, 0], err, None, 'Phase',None)
                plt.title(name)
                self.savefig(id + "corrollation_precent" + 'ph_' + name)
                

                self.err_plot(np.expand_dims(snrs, axis=[1])[:, 0], err, None, 'SNR', None)
                plt.title(name)
                self.savefig(id + "corrollation_precent" + 'snrs_' + name)

                j += 1
            errors_averaged.to_csv(self.saving_dir + id + "_errors_averaged.csv")


        # file.close()

    # %%
    def test_asig(self,shift_t, alpha_t, ph_t, nl):
        sns.set_style('white')
        id = "test_" + str(shift_t) + "_" + str(alpha_t) + "_" + str(nl)
        ampl_t = self.min_c + (self.max_c - self.min_c)/2 + np.multiply(np.random.random(size=(1, 1+self.numOfSig)), (self.max_c - self.max_c))
        y_n, y_wn = self.getSignal(ampl_t, shift_t, alpha_t, ph_t, nl,True)

        y_test_np = y_n.astype('complex64')
        y_test = torch.from_numpy(y_test_np[:, 0:self.truncSigLen])
        print(y_test.size())
        ampl, shift, damp, ph, y_out, _ = self.predict_ensembles(y_test)
        y_out_f = fft.fftshift(fft.fft(y_out, axis=2))
        y_out_mean = np.mean(y_out_f, 0).T
        y_n, y_wn, y_out_mean,y_out_f = y_n/50, y_wn/50, y_out_mean/50,y_out_f/50
        self.plotppm(fft.fftshift(fft.fft((y_n[:, 0]), axis=0)), 0, 5, False, linewidth=1, linestyle='-')
        self.plotppm(y_out_mean, 0, 5, False, linewidth=1, linestyle='--')
        self.plotppm(35 + (fft.fftshift(fft.fft((y_n[:, 0]), axis=0)) - np.squeeze(y_out_mean)), 0, 5, False, linewidth=1,
                linestyle='-')
        self.plotppm(37.5 +(fft.fftshift(fft.fft((y_wn[:, 0]), axis=0)) - np.squeeze(y_out_mean)), 0, 5, True, linewidth=1,
                linestyle='-')
        self.plot_basis(ampl/25, shift, damp,ph)
        # self.fillppm(30-2*np.std(y_out_f, 0).T, 30+2*np.std(y_out_f, 0).T, 0, 5, True, alpha=.1, color='red')
        y_f = fft.fftshift(fft.fft(y_n, axis=0), axes=0)
        plt.title(self.cal_snr(y_f))
        self.savefig(id + "_tstasig")
        
        # print(ampl_t - ampl)
        # print(np.sqrt(ampl_var))
        # print(self.cal_snrf(fft.fftshift(fft.fft(y_n))))

    # %%
    def monteCarlo(self,n, f, d, nl, ph,load):
        sns.set_style('whitegrid')
        id = "montc_" + str(f) + "_" + str(d) + "_" + str(ph) + "_" + str(nl) + "_" + str(n)
        ampl_t = (self.min_c) + (self.max_c - self.min_c) / 2 + np.multiply(np.random.random(size=(1, 1+self.numOfSig)), (self.max_c - self.max_c))
        if self.parameters['test_load']:
            y_test = np.load(os.path.join(self.sim_dir, id  + '.npy'))
        else:
            # ampl, shift, alpha, ph, noiseLevel
            y_n, y_wn = self.getSignal(ampl_t, f, d, ph, 0, True)
            noise = np.random.normal(0, nl, (n, self.sigLen)) + 1j * np.random.normal(0, nl, (n, self.sigLen))
            y_test = y_wn + noise.T
            # min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond
            np.save(self.sim_dir + id + '.npy', y_test)
            # np.savez(self.sim_dir + id, y_test)
            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), y_test,
                        self.sim_dir + id + '.txt')

        id = "mc/" + id + "/"
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        selected_met = ["Cr", "GPC", "Glu", "Ins", "NAA", "NAAG", "PCho", "PCr", "Tau"]
        df = pd.DataFrame(columns=['SNR', 'Frequency', 'Damping', 'Phase'])
        y_f = fft.fftshift(fft.fft(y_test, axis=0), axes=0)
        snrs = self.cal_snrf(y_f)
        df['SNR'] = snrs
        sns.histplot(snrs)
        self.savefig(id+"hist")
        y_test = y_test.astype('complex64')
        y_test = torch.from_numpy(y_test)
        print(y_test.size())
        # mean = [i+"_mean" for i in self.met_name]
        mean = self.met_name

        df[mean], df['Frequency'], df['Damping'], df['Phase'], decs, _, = self.predict_ensembles(y_test)
        cmap = 'Blues'
        sns.set_style('whitegrid')
        plot = sns.jointplot(x='Frequency', y='Damping', data=df,kind='kde',
                      cmap=cmap, marginal_kws={"shade": "True", "alpha": .2}, shade=True,
                      shade_lowest=False, alpha=.5)
        plot.ax_joint.axvline(f, color='gray', linestyle="--")
        plot.ax_joint.axhline(d, color='gray', linestyle="--")
        self.savefig(id + "mc_joint")
        


        for idx, name in enumerate(self.met_name):
            # sns.histplot(, kde=True)
            sns.distplot(df[name], hist=True,kde=False)
            sns.despine()
            plt.axvline(ampl_t[:, idx], color='k', linestyle="--")
            plt.title(self.met_name[idx])
            self.savefig(id + "_bias_" + name)
            
        dfm = df.melt("SNR")
        sns.set_style('whitegrid')
        g = sns.scatterplot(y='variable', x='value', data=dfm[dfm['variable'].isin(mean)], hue="variable", alpha=.1, zorder=1)
        # df = pd.DataFrame(y_out, columns=self.met_name)
        sns.despine()
        g = plt.scatter(y=mean, x=np.squeeze(ampl_t[:,0:len(mean)]), c='k')
        plt.legend([], [], frameon=False)
        self.savefig(id + "strip_plot")
        
        sns.set_style('whitegrid')
        corrmat = df.loc[:, mean].corr()
        sns.heatmap(corrmat, cmap=self.heatmap_cmap, linewidths=0.1)
        self.savefig(id + "mc_corr")

        
        corrmat = df.loc[:, mean].corr()
        g = sns.clustermap(corrmat, cmap=self.heatmap_cmap, linewidths=0.1)
        g.ax_row_dendrogram.remove()
        self.savefig(id + "cluster_mc_corr")
        
        x = abs(df.loc[:, mean] - ampl_t[:,0:len(mean)])
        x['SNR']  = df['SNR']
        sns.heatmap(x.corr(), cmap=self.heatmap_cmap, linewidths=0.1)
        self.savefig(id + "mc_ale_corr")
        
        df.to_csv(self.saving_dir+id+"_csvrslt")

    # %%
    def erroVsnoise(self,n, nl, f, d, ph,load):
        sns.set_style('whitegrid')
        id = "eVSn_" + str(f) + "_" + str(d) + "_" + str(ph) + "_" + str(nl) + "_" + str(n)
        ampl_t = (self.min_c) + (self.max_c - self.min_c) / 2 + np.multiply(np.random.random(size=(1, 1+self.numOfSig)), (self.max_c - self.max_c))
        if self.parameters['test_load']:
            y_test = np.load(os.path.join(self.sim_dir, id + 'errvsn_y_t' + '.npy'))
        else:
            # ampl, shift, alpha, ph, noiseLevel
            y_n, y_wn = self.getSignal(ampl_t, f, d, ph, nl, True)

            y_test = np.zeros((self.sigLen, n), dtype='complex128')
            nl_local = np.linspace(((5*nl)/n),nl,n)
            for i in range(0, n):
                noise = np.random.normal(0,nl_local[i] , (1, self.sigLen)) + 1j * np.random.normal(0, nl_local[i],
                                                                         (1, self.sigLen))
                y_test[:, i] = np.squeeze(y_wn + noise.T)
            np.save(os.path.join(self.sim_dir, id + 'errvsn_y_t' + '.npy'), y_test)

            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), y_test,
                        self.sim_dir + id + 'errorvsn_y_t.txt')

        id = "eVSn/" + id + "/"
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)

        y_test = y_test.astype('complex64')
        y_test = torch.from_numpy(y_test)
        df = pd.DataFrame(columns=['SNR', 'Frequency', 'Damping', 'Phase'])
        y_f = fft.fftshift(fft.fft(y_test, axis=0), axes=0)
        snrs = self.cal_snrf(y_f)
        df['SNR'] = snrs
        sns.histplot(snrs)
        self.savefig(id+"_hist")
        mean = self.met_name
        df[mean],  df['Frequency'], df['Damping'], df['Phase'], decs, _ = self.predict_ensembles(y_test)
        errs = np.abs(np.divide(np.squeeze(np.array(df[mean])) - ampl_t[:,0:len(mean)],ampl_t[:,0:len(mean)]))*100
        list_error = ["MAPE(" + str(i) + ")" for i in self.met_name]
        df[list_error] = errs
        cmap = 'Blues'
        for idx, name in enumerate(self.met_name):
            idi = "erroVsnoise_" + self.met_name[idx] + "_" + str(nl) + "_" + str(n) + str(f) + "_" + str(d)
            # ax = self.calib_plot(snrs, 100 * errs[:, idx], yerr, cmap)
            sns.regplot(x=df['SNR'],y=list_error[idx],data=df,order=2,ci=None)
            sns.despine()
            plt.title(self.met_name[idx])
            self.savefig(id + idi)

        list_corr_mean = mean + ['SNR', 'Frequency', 'Damping', 'Phase']
        #
        #
        # list_corr_mean = list_error + ['SNR']
        sns.heatmap(df[list_corr_mean].corr(),cmap=self.heatmap_cmap)
        self.savefig(id+"list_corr_mean")
        


        df["MAE(Frequency)"] = np.abs(((df['Frequency']) - f))
        df["MAE(Damping)"] = np.abs(((df['Damping']) - d))
        df["MAE(Phase)"] = np.abs(((df['Phase']) - ph)) * 180/np.pi
        dfm = df.loc[:, ["SNR", "MAE(Frequency)", "MAE(Damping)", "MAE(Phase)"]].melt('SNR', var_name='cols',
                                                                                         value_name='MAE')
        g = sns.lmplot(x="SNR", y="MAE", hue='cols', data=dfm, order=2, ci=None)
        sns.despine()
        self.savefig(id + "errors vs snr")
        df.to_csv(self.saving_dir + id + "_allrslt")

    # %%
    def testmodel(self, model, x):
        model.eval()
        with torch.no_grad():
            temp = model.forward(x)
        return temp
    def sigmoid(self,x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig
    def predict_ensembles(self,y_test, mean_= True):
        decs = []
        encs = []
        frl = []
        dampl = []
        phl = []
        sp = torch.nn.Softplus()
        for autoencoder in self.autoencoders:
            # expect dect, enct, enc, fr, damp, ph,mm_rec,dec
            dect, enct, _, fr, damp, ph, _, dec= self.testmodel(autoencoder,y_test.T.cuda())
            decs.append(dect.cpu().detach().numpy())
            encs.append(sp(enct).cpu().detach().numpy())
            frl.append(fr.cpu().detach().numpy())
            dampl.append(damp.cpu().detach().numpy())
            phl.append(ph.cpu().detach().numpy())
        if mean_:
            shift = (sum(frl) / len(frl))
            damp = (sum(dampl) / len(dampl))
            ph = (sum(phl) / len(phl))
        else:
            shift = np.asarray(frl)
            damp = np.asarray(dampl)
            ph = np.asarray(phl)
        encs_np = np.asarray((encs))
        if self.MM_type == 'single' or self.MM_type == 'single_param':
            ampl = np.concatenate((encs_np[:, :, 0:self.numOfSig],np.expand_dims(encs_np[:, :, 1*self.numOfSig],axis=2)), axis=2)
            if mean_:
                ampl = (np.mean(ampl, 0))
        else:
            ampl = encs_np[:, :, 0:self.numOfSig]
            if mean_:
                ampl = (np.mean(encs_np[:, :, 0:self.numOfSig], 0))

        return (ampl), shift, damp, ph, np.asarray(decs), encs_np[:, :, 0:self.numOfSig]

    def sp(self, x):
        return np.log(1 + np.exp(x))

    def dist_rslt(self,rslt):
        df = pd.DataFrame()
        npe = (rslt).cpu().detach().numpy()
        mm = npe[:, self.numOfSig:-3]
        mme = mm[:, 0:self.numOfMM]
        mme = self.sp(mm[:, 0:self.numOfMM])
        df = pd.DataFrame()
        df["mme_mean"] = mme.mean(axis=0)
        df["mme_std"] = mme.std(axis=0)
        mmf = mm[:, self.numOfMM:2 * self.numOfMM]
        df["mmf_mean"] = mmf.mean(axis=0)
        df["mmf_std"] = mmf.std(axis=0)
        df["mmd_mean"] = mm[:, -2].mean(axis=0)
        df["mmd_std"] = mm[:, -2].std(axis=0)
        df.to_csv(self.saving_dir + "dist_rslt.csv")

    def quantify(self):
        sns.set_style('white')
        path = "quantify/"
        Path(self.saving_dir+path).mkdir(exist_ok=True)
        data= np.load(self.saving_dir + "test_" + str(self.test_nos)+ ".npz")
        y_n, snr, idx = [data[x] for x in data]
        y_test_np = y_n.astype('complex64')
        y_test = torch.from_numpy(y_test_np[:, 0:self.truncSigLen])
        print(y_test.size())
        ampl, shift, damp, ph, y_out, _ = self.predict_ensembles(y_test,mean_=False)
        y_out_f = fft.fftshift(fft.fft(y_out, axis=2),axes=2)
        y_out_mean = np.mean(y_out_f, 0).T
        # y_n, y_wn, y_out_mean,y_out_f = y_n/50, y_wn/50, y_out_mean/50,y_out_f/50
        sns.set_palette('Set2')
        sns.set_style('white')
        # for ll in range(0,5):
        #     rng = range(ll*60,(ll+1)*64)
        #     self.plotsppm(fft.fftshift(fft.fft((y_n[0:1024, rng]), axis=0)), 0, 5, False, linewidth=0.3, linestyle='-')
        #     self.plotsppm(-30 + y_out_f[0, rng, 0:1024].T, 0, 5, False, linewidth=0.3, linestyle='--')
        #     self.plotsppm(+30 + (fft.fftshift(fft.fft((y_n[0:1024, rng]), axis=0)) - np.squeeze(y_out_f[0, rng, :].T)), 0, 5,
        #                   True,
        #                   linewidth=0.3,
        #                   linestyle='-')
        #     self.savefig(path + "subject_"+ str(ll) +"_tstasig")

        df_amplt = pd.DataFrame(ampl.squeeze(axis=0),columns=self.met_name)
        df_amplt['SNRs'] = snr
        df_amplt.to_csv(self.saving_dir + path + "result.csv")

        # self.plot_basis(ampl/25, shift, damp,ph)
        # self.fillppm(30-2*np.std(y_out_f, 0).T, 30+2*np.std(y_out_f, 0).T, 0, 5, True, alpha=.1, color='red')
        # y_f = fft.fftshift(fft.fft(y_n, axis=0), axes=0)
        # plt.title(self.cal_snr(y_f))
        rang = [0, 5]
        for ll in range(0,5):
            id = 64*ll+10

            self.plotppm(20 + np.fft.fftshift(np.fft.fft((y_test[0:1024, id])).T), rang[0], rang[1], False,
                         linewidth=0.3, linestyle='-')
            rec_signal, _, enc, fr, damp, ph, mm_v, _ = self.testmodel(self.autoencoders[0], y_test.T.cuda())
            self.plotppm(
                np.fft.fftshift(np.fft.fft(((mm_v).cpu().detach().numpy()[id, 0:self.truncSigLen])).T), rang[0],
                rang[1], False, linewidth=1, linestyle='-')
            self.plotppm(5 + np.fft.fftshift(np.fft.fft(
                (rec_signal.cpu().detach().numpy()[id, 0:self.truncSigLen])).T), rang[0], rang[1],
                         False, linewidth=1, linestyle='-')

            self.plotppm(50 + np.fft.fftshift(np.fft.fft(
                (y_test[0:self.truncSigLen, id] - rec_signal.cpu().detach().numpy()[id, 0:self.truncSigLen])).T),
                         rang[0], rang[1],
                         True, linewidth=1, linestyle='-')
            sns.despine()
            self.plot_basis((enc[id, :].unsqueeze(dim=0)).cpu().detach().numpy(), fr[id, :].cpu().detach().numpy(),
                            damp[id, :].cpu().detach().numpy(),
                            ph[id, :].cpu().detach().numpy())
            self.savefig(path +str(ll)+"_tstasig")



    def dotrain(self):
        """
        The function trains the autoencoder model.
        """
        if self.MM_plot == True:
            if 'param' in self.MM_type:
                mm = 0
                for idx in range(0, self.numOfMM):
                    if self.MM_conj == True:
                        x = np.conj(self.MM_model(self.MM_a[idx], 0, 0, 0, self.ppm2f(self.MM_f[idx]), self.MM_d[idx]))
                    else:
                        x = (self.MM_model(self.MM_a[idx], 0, 0, 0, self.ppm2f(self.MM_f[idx]), self.MM_d[idx]))
                    mm += x
                    if idx == self.numOfMM - 1:
                        self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, True)
                    else:
                        self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, False)
                self.savefig("MM")
                self.mm = mm.T
                Jmrui.write(Jmrui.makeHeader("tesDB", np.size(self.mm, 0), np.size(self.mm, 1), 0.25, 0, 0,
                                             1.2322E8), self.mm, self.saving_dir+'_mm.txt')

        if self.pre_plot == True:
            self.plot_basis2(self.basisset, 2)
        if self.tr is True:
            self.data_prep()
            autoencoders = []
            self.tic()
            for i in range(0,self.ens):
                pl.seed_everything(42)
                logger = TensorBoardLogger('tb-logs', name=self.loging_dir)
                lr_monitor = LearningRateMonitor(logging_interval='step')
                if self.parameters['early_stop'][0]:
                    early_stopping = EarlyStopping('val_acc',patience=self.parameters['early_stop'][1])
                    trainer= pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger,callbacks=[early_stopping,lr_monitor],accelerator='gpu',devices=self.parameters['gpu'])
                else:
                    trainer = pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger,callbacks=[lr_monitor],accelerator='gpu',devices=self.parameters['gpu'])
                # trainer= pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger,callbacks=[lr_monitor])
                logger.save()
                device = torch.device(self.parameters['gpu'])
                temp = Encoder_Model(self.depths[i], self.betas[i],self.reg_wei[i],self).to(device)
                x = summary(temp.met, (2, 1024))
                trainer.fit(temp, DataLoader(self.train, batch_size=self.batchsize,shuffle=True), DataLoader(self.val, batch_size=self.batchsize))
                autoencoders.append(temp)
                PATH = self.saving_dir + "model_"+ str(i) + ".pt"
                # Save
                torch.save(temp.state_dict(), PATH)
                del temp
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.memory_summary(device=None, abbreviated=False)
            self.toc("trining_time")
    def dotest(self):
        """
        It loads the trained models and evaluates them.
        """
        print("evaluation")
        self.autoencoders = []
        for i in range(0, self.ens):
            device = torch.device('cuda:0')
            model = Encoder_Model(self.depths[i], self.betas[i],self.reg_wei[i],self)
            PATH = self.saving_dir + "model_" + str(i) + ".pt"
            model.load_state_dict(torch.load(PATH, map_location=device))
            model.cuda()
            model.eval()
            self.autoencoders.append(model)
            # macs, params = profile(model.met, inputs=(torch.randn(1, 2, 1024),))
            # macs, params = get_model_complexity_info(model.met.encoder, (2, 1024), as_strings=True,
            #                                          print_per_layer_stat=True, verbose=True)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            x = summary(model.met,(2,1024))


        if self.sim_params is None:
            self.quantify()
        else:
            # self.test_compact()
            # plt.close()
            # self.test()
            # plt.close()
            # self.monteCarlo(256, 2, 5, 0.5, 0,True)
            # plt.close()
            # self.erroVsnoise(20, 1, 2, 5, 0,True)
            # plt.close()
            # self.test_asig(2, 5, 0, 0.5)
            plt.close()




    def tuner(self):
        """
        The function takes in the parameters of the model and the data, and then uses the Ray Tune library to find the best
        hyperparameters for the model.

        The function is called in the main function, and the best hyperparameters are then used to train the model.


        The function is called in the main
        """
        if self.MM_plot == True:
            if 'param' in self.MM_type:
                mm = 0
                for idx in range(0, self.numOfMM):
                    x = np.conj(self.MM_model(self.MM_a[idx], 0, 0, 0, self.ppm2f(self.MM_f[idx]), self.MM_d[idx]))
                    mm += x
                    if idx == self.numOfMM - 1:
                        self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, True)
                    else:
                        self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, False)
                self.savefig("MM")

                self.mm = mm.T
        if self.basis_need_shift[0] == True:
            self.basisset = self.basisset * np.exp(2 * np.pi * self.ppm2f(self.basis_need_shift[1]) * 1j * self.t)
        if self.pre_plot == True:
            self.plot_basis2(self.basisset, 2)
        self.data_prep()
        config = {
            # "lr": tune.loguniform(1e-5, 5e-4),
            "lr": 5e-5,
            "dp": tune.choice([2, 4, 6]),
            "batchsize" : tune.choice([32,128]),
            "reg_wei":tune.uniform(2, 7),
            "kw": tune.choice([5, 7]),
        }

        # config = {
        #     "lr":  1e-3,
        #     "dp": tune.choice([2, 4, 6]),
        #     "batchsize" : 64,
        #     "reg_wei":tune.uniform(2,7),
        #     "kw": tune.choice([5, 7]),
        # }
        # scheduler = ASHAScheduler(
        #     time_attr='training_iteration',
        #     max_t=self.max_epoch,
        #     grace_period=10,
        #     reduction_factor=3,
        #     brackets=1)
        algo = TuneBOHB(metric="mean_accuracy", mode="max")
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=self.max_epoch,
            stop_last_trials=False)
        # scheduler = PopulationBasedTraining(
        #     time_attr="training_iteration",
        #     perturbation_interval=10,
        #     burn_in_period=0,
        #     hyperparam_mutations={
        #         "lr": tune.loguniform(1e-5, 1e-2),
        #         "batchsize": [32,64,128],
        #         # "reg_wei": tune.uniform(2.0, 7.0),
        #     })
        reporter = CLIReporter(
            parameter_columns=["lr","dp","batchsize","reg_wei","kw"],
            metric_columns=["mean_accuracy", "training_iteration"])

        train_fn_with_parameters = tune.with_parameters(tunermodel, engine=self)
        resources_per_trial = {"cpu":10,"gpu": 1}
        # analysis = tune.run(
        #     train_fn_with_parameters,
        #     resources_per_trial={
        #         "cpu": 1,
        #         "gpu": 1
        #     },
        #     metric="mean_accuracy",
        #     mode="max",
        #     config=config,
        #     num_samples=100,
        #     scheduler=scheduler,
        #     progress_reporter=reporter,
        #     name="exp8_asa")
        analysis = tune.run(train_fn_with_parameters,
            num_samples=50,
            metric="mean_accuracy",
            mode="max",
            resources_per_trial=resources_per_trial,
            config=config,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="exp8_bhb",
            search_alg=algo)
        print("Best hyperparameters found were: ", analysis.best_config)

def tunermodel(config,engine=None):
    """
    It creates a PyTorch Lightning trainer, and then uses it to train a model

    :param config: a dictionary containing the hyperparameters to be tuned
    :param engine: the class that contains the data and the model
    """
    pl.seed_everything(42)
    data_dir = "~/data"
    logger = TensorBoardLogger(
        save_dir=tune.get_trial_dir(), name="", version=".")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if engine.parameters['early_stop'][0]:
        early_stopping = EarlyStopping('val_acc', patience=engine.parameters['early_stop'][1])
        trainer = pl.Trainer(gpus=1, max_epochs=engine.max_epoch, logger=logger,
                             progress_bar_refresh_rate=0,
                             callbacks=[early_stopping, lr_monitor,
                                        TuneReportCheckpointCallback(
                                            metrics={
                                                "mean_accuracy": "performance"
                                            },
                                            filename="checkpoint",
                                            on="validation_end")]
                                 #        TuneReportCallback(
                                 # {
                                 #     "mean_accuracy": "performance"
                                 # },
                                 # on="validation_end")]
                             )
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=engine.max_epoch, logger=logger,
                                progress_bar_refresh_rate=0,
                                callbacks=[lr_monitor,
                                           # TuneReportCheckpointCallback(
                                           #     metrics={
                                           #         "mean_accuracy": "performance"
                                           #     },
                                           #     filename="checkpoint",
                                           #     on="validation_end")])
                                           TuneReportCallback(
                                {
                                     "mean_accuracy": "performance"
                                 },
                                 on="validation_end")])
    # trainer= pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger,callbacks=[lr_monitor])
    logger.save()


    engine.parameters['lr'] = config['lr']
    engine.parameters['kw'] = config['kw']
    temp = Encoder_Model(config['dp'], 0, config['reg_wei'], engine)
    trainer.fit(temp.to(engine.parameters['gpu']), DataLoader(engine.train, batch_size=config['batchsize'],num_workers=255),
                DataLoader(engine.val, batch_size=config['batchsize'],num_workers=255))


