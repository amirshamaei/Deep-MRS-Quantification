import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import numpy as np
import scipy.io as sio
from torch import Tensor
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from Models.convnext import ConvNeXt
from Models.mlp_mixer_1d import MLPMixer1d
from Models.ConvNet import ConvNet
from Models.MLPNET import MLPNet


# The Encoder_Model class inherits from the LightningModule class, and it has a constructor that takes in a dictionary of
# hyperparameters
class Encoder_Model(pl.LightningModule):
    def __init__(self,depth, beta, tr_wei, param):
        """
        This function initializes the model, and sets up the parameters for the model.

        :param depth: number of layers in the network
        :param beta: the inverse temperature of the system
        :param tr_wei: weight of the regularization term
        :param param: the parameters of the model
        """
        super().__init__()
        self.param = param
        self.met = []
        self.selected_met = ["Cr", "GPC", "Ins", "NAA", "PCho", "Tau"]
        self.t = torch.from_numpy(param.t).float().cuda(self.param.parameters['gpu'])
        self.basis = torch.from_numpy(param.basisset[0:2048, 0:param.numOfSig].astype('complex64')).cuda(self.param.parameters['gpu'])
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False).cuda(self.param.parameters['gpu'])
        # self.max_c = torch.from_numpy(np.array(param.max_c)).float().cuda(self.param.parameters['gpu'])
        if self.param.MM_constr == True :
            print('tr is not in the model')
        else:
            self.tr = nn.Parameter(torch.tensor(0.004).cuda(self.param.parameters['gpu']), requires_grad=True)
        self.tr_wei = tr_wei
        self.act = nn.Softplus()
        self.lact = nn.ReLU6()
        self.sigm = nn.Sigmoid()
        self.model = None
        if param.enc_type == 'conv_simple':
            self.model = ConvNet
        if param.enc_type == 'mlp_simple':
            self.model = MLPNet
        # if param.enc_type == 'mlp_mixer':
        #     self.model = MLPMixer1d

        if self.param.MM == True :
            if self.param.MM_type == 'single' or self.param.MM_type == 'single_param':
                self.enc_out =  1 * (param.numOfSig+1) + 3*1
                self.mm = torch.from_numpy(param.mm[0:2048].astype('complex64')).cuda(self.param.parameters['gpu']).T
            if self.param.MM_type == 'param':
                if self.param.MM_fd_constr == False:
                    self.enc_out =  (1* (param.numOfSig) + 3*1)+(self.param.numOfMM*2)+1
                else:
                    self.enc_out = (1 * (param.numOfSig) + 3 * 1) + (self.param.numOfMM)

        else:
            self.enc_out =1 * (param.numOfSig) + 3
        # if self.param.MM == True :
        #     if self.param.MM_type == 'single' or self.param.MM_type == 'single_param':
        #         self.enc_out = 2 * (param.numOfSig + 1) + 3 * 1
        #         self.met = self.model(depth, param.banorm, self.enc_out)
        #         self.mm = torch.from_numpy(param.mm[0:2048].astype('complex64')).cuda().T
        #     if self.param.MM_type == 'param':
        #         self.MM_out = (self.param.numOfMM * 3)
        #         # if self.param.MM_constr == True:
        #         #     self.MM_out += 1
        #         self.enc_out = 2 * (param.numOfSig) + 3 * 1
        #         self.met = self.model(depth, param.banorm, self.enc_out)
        #         self.MM_net = self.model(depth, param.banorm, self.MM_out)
        # else:
        #     self.MM_out = 2 * (param.numOfSig) + 3
        if self.param.MM_constr == True:
            self.enc_out += 1
        if param.enc_type == 'convnext_tiny':
            self.met = ConvNeXt(num_classes=self.enc_out)
        elif param.enc_type == 'mlp_mixer':
            self.met = MLPMixer1d(in_channels=2, image_size=self.param.truncSigLen, patch_size=16, z_dim=self.enc_out,
                                         dim=512, depth=8, token_dim=256, channel_dim=2048)
        else:
            self.met = self.model(depth, param.banorm, self.enc_out,self.param.parameters['kw'])
        if param.parameters['MM_model'] == "lorntz":
            self.MM_model = self.Lornz
        if param.parameters['MM_model'] == "gauss":
            self.MM_model = self.Gauss

    def sign(self,t,eps):
        """
        sgn function
        It takes in a tensor t and a small number eps, and returns a tensor of the same shape as t

        :param t: the input tensor
        :param eps: This is the epsilon value used in the sign function
        :return: The sign of the tensor t.
        """
        return (t/torch.sqrt(t**2+ eps))

    def Gauss(self, ampl, f, d, ph, Crfr, Cra, Crd):
        """
        It returns a complex-valued Gaussian function with amplitude, frequency, decay, phase, frequency reference,
        amplitude reference, and decay reference parameters

        :param ampl: Amplitude of the Gaussian
        :param f: frequency
        :param d: decay
        :param ph: phase
        :param Crfr: Carrier frequency
        :param Cra: Amplitude of the Gaussian
        :param Crd: The decay rate of the Gaussian
        :return: the complex-valued Gaussian waveform.
        """
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(d+Crd)**2 * self.t.T*self.t.T))
    def Lornz(self, ampl, f, d, ph, Crfr, Cra, Crd):
        """
        It returns a complex exponential with a frequency of $f + Crfr$, a phase of $ph$, an amplitude of $Cra*ampl$, and a
        decay of $d+Crd$

        :param ampl: Amplitude of the signal
        :param f: frequency
        :param d: decay
        :param ph: phase
        :param Crfr: Carrier frequency
        :param Cra: Amplitude of the carrier wave
        :param Crd: decay rate
        :return: the Lornz function with the parameters that are passed in.
        """
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(d+Crd) * self.t.T))
    def Voigt(self, ampl, f, dl,dg, ph, Crfr, Cra, Crd):
        """
        The function takes in a set of parameters and returns a complex valued Voigt function

        :param ampl: Amplitude of the signal
        :param f: frequency
        :param dl: Lorentzian linewidth
        :param dg: Gaussian decay
        :param ph: phase
        :param Crfr: frequency shift
        :param Cra: Amplitude of the real part of the signal
        :param Crd: Drift
        :return: the complex value of the Voigt function.
        """
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(((dl) * self.t.T)+(dg+Crd) * self.t.T*self.t.T)))
    def forward(self, x):
        """
        The function takes in a signal, passes it through a neural network, and returns the signal that the model-decoder
        predicts

        :param x: input signal
        :return: The output of the forward function is the reconstructed signal, the encoded signal, the encoded signal
        without the MM, the frequency, the damping, the phase, the reconstructed MM signal, and the reconstructed signal
        without the MM.
        """
        mm_rec = 0
        enct = self.met(self.param.inputSig(x))

        enc = self.act(enct[:, 0:(self.param.numOfSig)])
        # enc = self.sigm(enc)
        # fr = self.reparameterize(torch.unsqueeze(enct[:, -5],1),torch.unsqueeze(enct[:, -6],1))
        # damp = self.reparameterize(torch.unsqueeze(enct[:, -3],1),torch.unsqueeze(enct[:, -4],1))
        # ph = self.reparameterize(torch.unsqueeze(enct[:, -1],1),torch.unsqueeze(enct[:, -2],1))
        fr = torch.unsqueeze(enct[:, -3],1)
        damp = torch.unsqueeze(enct[:, -2],1)
        ph = torch.unsqueeze(enct[:, -1],1)
        # dec = self.decoder(enc)
        sSignal = torch.matmul(enc[:, 0:(self.param.numOfSig)] + 0 * 1j, self.basis.T)
        dec = torch.multiply(sSignal, torch.exp(-2 * math.pi * (fr) * self.t.T * 1j))
        dec = torch.multiply(dec, torch.exp((-1 * damp) * self.t.T))
        dec = (dec * torch.exp(ph * 1j))

        if (self.param.MM == True):
            if self.param.MM_type == 'single' or self.param.MM_type == 'single_param':
                mm_enct = enct[:, ((self.param.numOfSig)):-3]
                mm_enc = self.act(mm_enct[:, 0])
                mm_rec = (mm_enc[:].unsqueeze(1)) * self.mm
                mm_rec = torch.multiply(mm_rec, torch.exp(-2 * math.pi * (fr) * self.t.T * 1j))
                mm_rec = torch.multiply(mm_rec, torch.exp((-1 * damp) * self.t.T))

            if self.param.MM_type == 'param':
                if self.param.MM_fd_constr == False:
                    mm_enct = enct[:, ((self.param.numOfSig)):-3]
                    mm_enc = self.act(mm_enct[:, 0:(self.param.numOfMM)])
                    mm_f = mm_enct[:, (self.param.numOfMM):(self.param.numOfMM)*2]
                    mm_damp = (mm_enct[:, -2])

                    for idx in range(0, len(self.param.MM_f)):
                        mm_rec += self.MM_model((mm_enc[:, idx].unsqueeze(1)), torch.unsqueeze(mm_f[:,idx],1),
                                                torch.unsqueeze(mm_damp,1), torch.tensor(0), self.param.trnfreq * (self.param.MM_f[idx]),
                                                self.param.MM_a[idx], self.param.MM_d[idx])
                else:
                    mm_enct = enct[:, ((self.param.numOfSig)):-3]
                    mm_enc = self.act(mm_enct[:, 0:(self.param.numOfMM)])
                    # mm_enc = self.reparameterize(mm_enct[:, 0:(self.param.numOfMM)],
                    #                          mm_enct[:, (self.param.numOfMM):(self.param.numOfMM) * 2])
                    for idx in range(0, len(self.param.MM_f)):
                        mm_rec += self.MM_model((mm_enc[:,idx].unsqueeze(1)), fr,
                                                    damp, torch.tensor(0), self.param.trnfreq * (self.param.MM_f[idx]),
                                                      self.param.MM_a[idx], self.param.MM_d[idx])
                if self.param.MM_conj:
                    mm_rec = torch.conj(mm_rec)
            if self.param.MM_constr == True :
                self.tr = (self.sigm(mm_enct[:,-1]-5))
        mm_rec = mm_rec * torch.exp(ph * 1j)
        dect = dec + mm_rec
        return dect, enct, enc, fr, damp, ph,mm_rec,dec

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        "Given a mean and standard deviation, sample from a Gaussian distribution."

        The reparameterization trick is a way to sample from a Gaussian distribution without having to explicitly sample
        from a Gaussian distribution.

        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :type mu: Tensor
        :param logvar: (Tensor) log variance of the latent Gaussian [B x D]
        :type logvar: Tensor
        :return: A tensor of size [B x D]
        """
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[4]
        ampl_b = args[3]
        bat_idx = args[5]
        mm = args[6]
        # Account for the minibatch samples from the dataset
        # cut_signal = args[7]
        # cut_dec =args[8]
        met_loss = 0
        loss_real = 0
        loss_imag = 0
        if (self.param.MM == True) and  (self.param.MM_constr == True):
            cut_signal = (((1 + self.sign(self.t - self.tr, 0.0000001)) / 2)[0:self.param.truncSigLen].T * input[:, 0:self.param.truncSigLen])
            cut_dec = (((1 + self.sign(self.t - self.tr, 0.0000001)) / 2).T * recons.clone())
            loss_real = self.criterion(cut_dec.real[:, 0:self.param.truncSigLen],
                                       cut_signal.real[:, 0:self.param.truncSigLen])
            loss_imag = self.criterion(cut_dec.imag[:, 0:self.param.truncSigLen],
                                       cut_signal.imag[:, 0:self.param.truncSigLen])
            met_loss = (loss_real + loss_imag) / (2 * self.param.truncSigLen)
            self.log("met_loss", met_loss)
            # reg = (tri / (self.param.truncSigLen))
            self.tr_ = torch.mean(self.tr)
            self.log("reg", self.tr_)
            met_loss= (met_loss + (self.tr_) * self.tr_wei * (self.param.batchsize))*0.5
            self.log("train_los", met_loss)

        recons += mm
        init_point = 1
        loss_real = self.criterion(recons.real[:, init_point:self.param.truncSigLen], input.real[:, init_point:self.param.truncSigLen])
        loss_imag = self.criterion(recons.imag[:, init_point:self.param.truncSigLen], input.imag[:, init_point:self.param.truncSigLen])
        recons_loss = (loss_real+loss_imag)/(2*self.param.truncSigLen)
        self.log("recons_loss", recons_loss)
        # self.log("mm_loss", mm_loss)
        loss = met_loss + recons_loss
        return loss,recons_loss

    def training_step(self, batch, batch_idx):
        """
        The function takes in a batch of data, and returns a dictionary of losses

        :param batch: the batch of data
        :param batch_idx: the index of the batch
        :return: The loss and the reconstruction loss
        """
        if self.param.sim_params is None:
            x = batch[0]
        else:
            x, ampl_batch = batch[0],batch[1]
        dec_real, enct,enc,_,_,_,mm,dec = self(x)
        logvar = enct[:, (self.param.numOfSig):2*(self.param.numOfSig)]

        loss_mse,recons_loss = [lo/len(x) for lo in self.loss_function(dec, x, enct[:, 0:(self.param.numOfSig)],_,_,batch_idx,mm)]

        return {'loss': loss_mse,'recons_loss':recons_loss}



    def validation_step(self, batch, batch_idx):
        """
        It takes a batch of data, and returns the r2 value for the batch

        :param batch: The batch of data passed to the validation_step function
        :param batch_idx: The index of the batch within the current epoch
        :return: The validation step returns the r2 value.
        """
        r2 = 0
        if self.param.sim_params is None:
            x = batch[0]
        else:
            x, label = batch[0], batch[1]
            ampl_batch, alpha_batch = label[:,0:-2],label[:,-1]
            _, _, enc, _, alpha, _, mm, dec = self(x)
            error = (ampl_batch[:,0:self.param.numOfSig] - enc[:,:])
            mean = torch.mean(ampl_batch[:,0:self.param.numOfSig],dim=0)
            stot = (ampl_batch[:,0:self.param.numOfSig] - mean)
            r2 = 1-(torch.sum(error**2,dim=0)/torch.sum(stot**2,dim=0))
        # r2 = torch.sum(r2)
        # self.log("r2_total", r2)
        # error = (ampl_batch[:,12] - enc[:,12])
        # mean = torch.mean(ampl_batch[:,12],dim=0)
        # stot = (ampl_batch[:,12] - mean)
        # r2_naa = 1-(torch.sum(error**2)/torch.sum(stot**2))
        # error = (alpha_batch - torch.squeeze(alpha))
        # mean = torch.mean(alpha_batch,dim=0)
        # stot = (alpha_batch - mean)
        # r2_alpha = 1-(torch.sum(error**2)/torch.sum(stot**2))
        # self.log("r2",r2_naa)
        results = self.training_step(batch, batch_idx)
        if (self.current_epoch % self.param.parameters['val_freq'] == 0 and batch_idx == 0):
            id = np.int(np.random.rand() * 300)
            # sns.scatterplot(x=alpha_batch.cpu(), y=error.cpu())
            # sns.scatterplot(x=10*ampl_batch[:,12].cpu(),y=10*enc[:,12].cpu())
            # # plt.title(str(r2))
            # plt.show()
            # ampl_t = min_c + np.multiply(np.random.random(size=(1, 21)), (max_c - max_c))
            # y_n, y_wn = getSignal(ampl_t, 0, 5, 0, 0.5)
            rang = [0, 5]
            # id= 10
            # plotppm(np.fft.fftshift(np.fft.fft((y_n.T)).T), 0, 5,False, linewidth=0.3, linestyle='-')
            self.param.plotppm(np.fft.fftshift(np.fft.fft((self.param.y_test_trun[id, :])).T), rang[0], rang[1], False, linewidth=0.3, linestyle='-')
            # plt.plot(np.fft.fftshift(np.fft.fft(np.conj(y_trun[id, :])).T)[250:450], linewidth=0.3)
            rec_signal,_,enc, fr, damp, ph,mm_v,_ = self(torch.unsqueeze(self.param.y_test_trun[id, :], 0).cuda())
            # plotppm(np.fft.fftshift(np.fft.fft(((rec_signal).cpu().detach().numpy()[0,0:truncSigLen])).T), 0, 5,False, linewidth=1, linestyle='--')
            if self.param.MM == True:
                self.param.plotppm(np.fft.fftshift(np.fft.fft(((mm_v).cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T), rang[0], rang[1], False, linewidth=1,linestyle='--')
            self.param.plotppm(np.fft.fftshift(np.fft.fft(
                (rec_signal.cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T), rang[0], rang[1],
                    False, linewidth=1, linestyle='--')

            self.param.plotppm(20 + np.fft.fftshift(np.fft.fft(
                (self.param.y_test_trun[id, :]-rec_signal.cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T), rang[0], rang[1],
                    True, linewidth=1, linestyle='--')
            sns.despine()
            self.param.plot_basis((enc).cpu().detach().numpy(), fr.cpu().detach().numpy(), damp.cpu().detach().numpy(), ph.cpu().detach().numpy())
            # plt.plot(np.fft.fftshift(np.fft.fft(np.conj(rec_signal.cpu().detach().numpy()[0,0:trunc])).T)[250:450], linewidth=1,linestyle='--')
            plt.title("#Epoch: " + str(self.current_epoch))
            self.param.savefig(self.param.epoch_dir+"paper1_1_epoch_" + str(self.current_epoch) +"_"+ str(self.tr_wei))

        self.log("val_acc", results['loss'])
        self.log("val_recons_loss", results['recons_loss'])
        return r2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.param.parameters['lr'])
        # if self.param.parameters['reduce_lr'][0] ==True:
        lr_scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=self.param.parameters['reduce_lr'][0]),
            # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5),
            # 'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99),
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=[int(self.param.max_epoch/8),int(self.param.max_epoch/2)],
                                                              gamma=self.param.parameters['reduce_lr'][0]),
            # 'monitor':self.param.parameters['reduce_lr'][1],
            'name': 'scheduler'
        }
        return [optimizer],[lr_scheduler]

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("epoch_los",avg_loss)

    def validation_epoch_end(self,validation_step_outputs):
        """
        It takes the output of the validation step, which is a list of lists, and calculates the mean of each list, which is
        the r2 for each metabolite.

        :param validation_step_outputs: a list of outputs from the validation step
        """
        r2 = []
        # naa = []
        # alpha = []
        for list in validation_step_outputs:
            # naa.append(list[0])
            # alpha.append(list[1])
            r2.append((list))
        try:
            r2 = torch.mean(torch.stack(r2),axis=0)
            # r2_naa = torch.mean(torch.stack(naa))
            # self.log("r2_naa", r2_naa)
            # r2_alpha = torch.mean(torch.stack(alpha))
            # self.log("r2_alpha", r2_alpha)
            performance = 0
            for idx,name in enumerate(self.param.met_name[:-1]):
                self.log(name,r2[idx])
            for name in self.selected_met:
                performance+=r2[self.param.met_name.index(name)]
            performance=performance/len(self.selected_met)
            self.log("performance",performance)
            r2_total = torch.mean(r2)
            self.log("r2_total",r2_total)
        except:
            pass
            # print("r2 cannot be calculated")
