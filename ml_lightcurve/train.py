"""
# Create Neural Network

### Primary Sources:
- [Paper by Lukosiute](https://arxiv.org/pdf/2204.00285.pdf) with [GitHub code](https://github.com/klukosiute/kilonovanet) using [Bulla's data](https://github.com/mbulla/kilonova_models/tree/master/bns/bns_grids/bns_m3_3comp).
- [PELS-VAE](https://github.com/jorgemarpa/PELS-VAE) github that had was used to draft train part for Lukosiute net ([data for it](https://zenodo.org/records/3820679#.XsW12RMzaRc))

### Secondary Sources
- [Tronto Autoencoder](https://www.cs.toronto.edu/~lczhang/aps360_20191/lec/w05/autoencoder.html) (Convolutional net)
- [Video with derivations](https://www.youtube.com/watch?v=iwEzwTTalbg)
- [Data sampling with scikit](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)
- [Astro ML BOOK repo with code](https://github.com/astroML/astroML_figures/blob/742df9181f73e5c903ea0fd0894ad6af83099c96/book_figures/chapter9/fig_sdss_vae.py#L45)

"""
import numpy as np
import h5py
import os
import datetime
import pandas as pd

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing

from model_cvae import CVAE

class LightCurveDataset(Dataset):
    """
    LightCurve dataset
    Dispatches a lightcurve to the appropriate index
    """
    def __init__(self, pars:np.ndarray, lcs:np.ndarray, times:np.ndarray, device="cuda"):
        self.device = device
        self.pars = np.array(pars)
        self.lcs = np.array(lcs)
        assert self.pars.shape[0] == self.lcs.shape[0], "size mismatch between lcs and pars"
        self.times = times
        self.len = len(self.lcs)

        # preprocess parameters
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(pars)
        self.pars_normed = self.scaler.transform(pars)
        # inverse transform
        # inverse = scaler.inverse_transform(normalized)
        if np.min(self.pars_normed) < 0 or np.max(self.pars_normed) > 1:
            raise ValueError("Parameter normalization error")
        # preprocess lcs
        self._transform_lcs(self.lcs)
        if np.min(self.lcs_log_norm) < 0 or np.max(self.lcs_log_norm) > 1:
            raise ValueError("Parameter normalization error")

    def __getitem__(self, index):
        """ returns image/lc, vars(params)[normalized], vars(params)[physical] """
        return (torch.from_numpy(self.lcs_log_norm[index]).to(self.device).to(torch.float),
                torch.from_numpy(self.pars_normed[index]).to(self.device).to(torch.float),  # .reshape(-1,1)
                self.lcs[index],
                self.pars[index])

    def __len__(self):
        return len(self.lcs)

    def _transform_lcs(self, lcs):
        log_lcs = np.log10(lcs)
        self.lc_min = log_lcs.min()
        self.lc_max = log_lcs.max()
        self.lcs_log_norm = (log_lcs - np.min(log_lcs)) / (np.max(log_lcs) - np.min(log_lcs))

    def _transform_pars(self, _pars):
        print(_pars.shape, self.pars[:,0].shape)
        for i, par in enumerate(_pars.flatten()):
            if (par < self.pars[:,i].min()) or (par > self.pars[:,i].max()):
                raise ValueError(f"Parameter '{i}'={par} is outside of the training set limits "
                                 f"[{self.pars[:,i].min()}, {self.pars[:,i].max()}]")
        return self.scaler.transform(_pars)

    def inverse_transform_lc_log(self, lcs_log_normed):
        return np.power(10, lcs_log_normed * (self.lc_max - self.lc_min) + self.lc_min)

    def get_dataloader(self, batch_size=32, test_split=0.2):
        """
        If
        :param batch_size: if 1 it is stochastic gradient descent, else mini-batch gradient descent
        :param test_split:
        :return:
        """
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(self, batch_size=batch_size,
                                  sampler=train_sampler, drop_last=False)
        test_loader = DataLoader(self, batch_size=batch_size,
                                 sampler=test_sampler, drop_last=False)

        return (train_loader, test_loader)

# Initialize learning Rate scheduler
def select_scheduler(optimizer, lr_sch='step')->optim.lr_scheduler or None:
    if lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=10,
                                              gamma=0.1 # gamma = decaying factor
                                              )
    elif lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=0.985)
    elif lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=50,
                                                         eta_min=1e-5)
    elif lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=.5,
                                                         verbose=True)
    else:
        scheduler = None
    return scheduler

class EarlyStopping:
    """Early stops the training if validation loss doesn't
    improve after a given patience."""
    def __init__(self, patience=7, min_delta=0., verbose=False):
        """
        Attributes
        ----------
        patience  : int
            How long to wait after last time validation loss improved.
            Default: 7
        min_delta : float
            Minimum change in monitored value to qualify as
            improvement. This number should be positive.
            Default: 0
        verbose   : bool
            If True, prints a message for each validation loss improvement.
            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_delta = min_delta

    def __call__(self, val_loss):

        current_loss = val_loss

        if self.best_score is None:
            self.best_score = current_loss
        elif torch.abs(current_loss - self.best_score) < self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_loss
            self.counter = 0

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

class Trainer:
    def __init__(self, model:CVAE, optimizer, batch_size, scheduler, beta, print_every, device):
        self.device = device
        self.model = model
        if torch.cuda.device_count() > 1 and True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        print('Is model in cuda? ', next(self.model.parameters()).is_cuda)
        self.opt = optimizer
        self.sch = scheduler
        self.batch_size = batch_size
        self.train_loss = {
            'KL_latent': [], 'BCE': [], 'Loss': [],
            'MSE': [], 'KL_output': [], 'tMSE': [],
            'wMSE': [], 'beta': []
        }
        self.test_loss = {
            'KL_latent': [], 'BCE': [], 'Loss': [],
            'MSE': [], 'KL_output': [], 'tMSE': [],
            'wMSE': [], 'beta': []
        }
        self.num_steps = 0
        self.print_every = print_every
        self.beta = beta

        # ---
        self.model_dir = os.getcwd() + '/models/'
        self.run_name = "test"

    def train(self, train_loader, test_loader, early_stopper, epochs, save=True, save_chkpt=False, early_stop=False):
        # hold samples, real and generated, for initial plotting

        # train for n number of epochs
        time_start = datetime.datetime.now()
        train_loss = []
        test_loss = []

        total_train_losses = { key : [] for key in self.train_loss.keys()}
        total_test_losses  = { key : [] for key in self.test_loss.keys()}

        for epoch in range(epochs):
            # one complete pass of the entire training dataset through the learning algorithm
            e_time = datetime.datetime.now()
            print("\n")
            print(f"========================| Epoch {epoch}/{epochs} |========================")

            # train and validate
            self._train_epoch(train_loader, epoch)

            # test and get the final loss
            val_loss = self._test_epoch(test_loader, epoch)

            # update learning rate according to scheduler
            if self.sch is not None:
                if 'ReduceLROnPlateau' == self.sch.__class__.__name__:
                    self.sch.step(val_loss)
                else:
                    self.sch.step(epoch)

            # report elapsed time per epoch and total run time
            epoch_time = datetime.datetime.now() - e_time
            elap_time = datetime.datetime.now() - time_start

            # compute normalized losses across batches
            # idx1 = epoch*self.batch_size  + 1
            # idx2 = (epoch+1)*self.batch_size + 1

            # store total losses
            for key in self.train_loss.keys():
                total_train_losses[key].append( np.sum(self.train_loss[key][-self.batch_size:]) )
                total_test_losses[key].append( np.sum(self.test_loss[key][-self.batch_size:]) )

            print(f"T={elap_time.seconds/60:.2f} m | T/Ep={epoch_time.seconds:.2f}s | "
                  f"TrainLoss={total_train_losses['Loss'][-1]:.4f} | TestLoss={total_test_losses['Loss'][-1]:.4f}")

            # save model state at a given epoch
            if save_chkpt:
                print("Saving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    # 'train_loss': train_running_loss,
                    # 'val_loss': val_running_loss,
                    'beta':self._beta_scheduler(epoch)
                },
                    '%s/VAE_model_%s_%d.chkpt' % (self.model_dir, self.run_name, epoch))

            # stop training if a condition is met
            if early_stop:
                early_stopper(val_loss.cpu())
                if early_stopper.early_stop:
                    print("Early stopping")
                    break

        # save final model state
        if save:
            fname = '%s/VAE_model_%s.pt' % (self.model_dir, self.run_name)
            print(f"Saving model {fname}")
            torch.save(self.model.state_dict(),fname)
            print(f"Model saved: {fname}")

            out_train = pd.DataFrame.from_dict(total_train_losses)
            out_train.to_csv(fname.replace('.pt','_train_losses.csv'), index=False)
            out_test = pd.DataFrame.from_dict(total_test_losses)
            out_test.to_csv(fname.replace('.pt','_test_losses.csv'), index=False)
            print(f"Train/test losses saved")

        return (train_loss, test_loss)

    def _test_epoch(self, test_loader, epoch):
        """Testing loop for a given epoch.

        Parameters
        ----------
        data_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number

        Returns
        -------

        """

        # set model in an 'eval' mode
        self.model.eval()
        # running_loss = 0
        with torch.no_grad():
            xhat_plot, x_plot, l_plot = [], [], []
            for i, (data, label, data_phys, label_phys) in enumerate(test_loader):
                # move data to device where model is
                data = data.to(self.device)
                label = label.to(self.device)
                # evaluate model on the data
                xhat, mu, logvar, z = self.model(data, label)
                # compute loss
                loss = self._loss(data, xhat, mu, logvar, train=False, ep=epoch)
                # save batch loss
                # running_loss += loss.item() * data.size(0)

        self._report_test(epoch)

        return loss

    def _train_epoch(self, data_loader, epoch) -> None:
        """
        Training loop for a given epoch. It goes over
        batches, light curves

        Parameters
        ----------
        data_loader : pytorch object
            data loader object with training items
        epoch       : int
            epoch number

        Returns
        -------
        """

        # set model into training mode
        self.model.train()

        # iterate over len(data)/batch_size
        mu_ep, labels = [], []
        xhat_plot, x_plot, l_plot = [], [], []

        # Get data for [start:start+batch_size] for each epoch
        for i, (data, label, data_phys, label_phys) in enumerate(data_loader):
            self.num_steps += 1
            # mode train data to device
            data = data.to(self.device)
            label = label.to(self.device)

            # Resets the gradients of all optimized tensors (Clear gradients w.r.t. parameters)
            self.opt.zero_grad()

            # Forward pass only to get logits/output (evaluate model)
            xhat, mu, logvar, z = self.model(data, label)

            # compute/store loss
            loss = self._loss(data, xhat, mu, logvar, train=True, ep=epoch)

            # computes dloss/dx for every parameter x which has requires_grad=True.
            loss.backward()

            # perform a single optimization step
            self.opt.step()

            # print train loss
            self._report_train(i)

    def _loss(self, x, xhat, mu, logvar, train=True, ep=0):
        """
        Evaluates loss function and add reports to the logger.
        Loss function is weighted MSe + KL divergeance. Also BCE
        is calculate for comparison.

        Parameters
        ----------
        x      : tensor
            tensor of real values
        xhat   : tensor
            tensor of predicted values
        mu     : tensor
            tensor of mean values
        logvar : tensor
            tensor of log vairance values
        train  : bool
            wheather is training step or not
        ep     : int
            epoch value of training loop

        Returns
        -------
        loss
            loss value
        """

        bce = F.binary_cross_entropy(xhat, x, reduction='mean')

        mse = F.mse_loss(xhat, x, reduction='mean')

        wmse = torch.sum((xhat - x) ** 2 / x ** 2) / (x.shape[0] * x.shape[1])

        kld_l = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp()) / x.shape[0] #/ 1e4

        kld_o = -1. * F.kl_div(xhat, x, reduction='mean')

        # beta is probably https://paperswithcode.com/method/beta-vae#:~:text=Beta%2DVAE%20is%20a%20type,independence%20constraints%20with%20reconstruction%20accuracy.
        beta = self._beta_scheduler(ep)

        # compute total loss
        loss = bce + beta * kld_l #+ 1 * kld_o
        # loss = mse + beta * kld_o

        # log the result
        if train:
            self.train_loss['BCE'].append(bce.item())
            self.train_loss['MSE'].append(mse.item())
            self.train_loss['wMSE'].append(wmse.item())
            self.train_loss['KL_latent'].append(kld_l.item())
            self.train_loss['KL_output'].append(kld_o.item())
            self.train_loss['beta'].append(beta.item())
            self.train_loss['Loss'].append(loss.item())
        else:
            self.test_loss['BCE'].append(bce.item())
            self.test_loss['MSE'].append(mse.item())
            self.test_loss['wMSE'].append(wmse.item())
            self.test_loss['KL_latent'].append(kld_l.item())
            self.test_loss['KL_output'].append(kld_o.item())
            self.test_loss['beta'].append(beta.item())
            self.test_loss['Loss'].append(loss.item())
        return loss

    def _report_train(self, i):
        """Report training metrics to logger and standard output.

        Parameters
        ----------
        i : int
            training step

        Returns
        -------

        """
        if (i % self.print_every == 0):
            print("\t Training iteration/batch %i, global step %i" % (i + 1, self.num_steps))
            print("\t BCE  : %3.4f" % (self.train_loss['BCE'][-1]))
            print("\t MSE  : %3.4f" % (self.train_loss['MSE'][-1]))
            print("\t wMSE : %3.4f" % (self.train_loss['wMSE'][-1]))
            print("\t KL_l : %3.4f" % (self.train_loss['KL_latent'][-1]))
            print("\t KL_o : %3.4f" % (self.train_loss['KL_output'][-1]))
            print("\t beta : %3.4f" % (self.train_loss['beta'][-1]))
            print("\t Loss : %3.4f" % (self.train_loss['Loss'][-1]))
            print("-"*20)

    def _report_test(self, ep):
        """Report testing metrics to logger and standard output.

        Parameters
        ----------
        i : int
            training step

        Returns
        -------

        """
        print('*** TEST LOSS ***')
        print("Epoch %i, global step %i" % (ep, self.num_steps))
        print("\t BCE  : %3.4f" % (self.train_loss['BCE'][-1]))
        print("\t MSE  : %3.4f" % (self.train_loss['MSE'][-1]))
        print("\t wMSE : %3.4f" % (self.train_loss['wMSE'][-1]))
        print("\t KL_l : %3.4f" % (self.train_loss['KL_latent'][-1]))
        print("\t KL_o : %3.4f" % (self.train_loss['KL_output'][-1]))
        print("\t beta : %3.4f" % (self.train_loss['beta'][-1]))
        print("\t Loss : %3.4f" % (self.train_loss['Loss'][-1]))
        print("-"*20)

    def _beta_scheduler(self, epoch, beta0=0., step=50, gamma=0.1):
        """Scheduler for beta value, the sheduler is a step function that
        increases beta value after "step" number of epochs by a factor "gamma"

        Parameters
        ----------
        epoch : int
            epoch value
        beta0 : float
            starting beta value
        step  : int
            epoch step for update
        gamma : float
            linear factor of step scheduler

        Returns
        -------
        beta
            beta value
        """
        if self.beta == 'step':
            return np.float32( beta0 + gamma * (epoch+1 // step) )
        else:
            return np.float32(self.beta)

def train_main(lr=0.1, batch_size=64):
    # Load Prepared data
    with h5py.File(os.getcwd()+'/data/'+"X.h5","r") as f:
        lcs = np.array(f["X"])
        times = np.array(f["time"])
    with h5py.File(os.getcwd()+'/data/'+"Y.h5","r") as f:
        pars = np.array(f["Y"])
        features_names = list([str(val.decode("utf-8")) for val in f["keys"]])

    print(f"lcs={lcs.shape}, pars={pars.shape}, times={times.shape}")
    print(f"lcs={lcs.min()}, {lcs.max()}, pars={pars.min()} {pars.max()}, times={times.shape}")
    print(features_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()


    # init dataloaders for training
    dataset = LightCurveDataset(pars, lcs, times)
    # create data loaders (feed data to model for each fold)
    train_loader, test_loader = dataset.get_dataloader(batch_size=batch_size, test_split=.2)


    # init model
    model = CVAE(image_size=len(lcs[0]),  #  150
                 hidden_dim=700,
                 z_dim=4 * len(features_names),
                 c=len(features_names))
    model.to(device)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Num of trainable params: {n_train_params}')
    # print(model)

    # Initialize optimizers
    # lgorithms that adjust the model's parameters during training to minimize a loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    # '''
    # lr = lr * factor
    # mode='max': look for the maximum validation accuracy to track
    # patience: number of epochs - 1 where loss plateaus before decreasing LR
    #         # patience = 0, after 1 bad epoch, reduce LR
    # factor = decaying factor
    scheduler = select_scheduler(optimizer, lr_sch='step')

    print('Optimizer    :', optimizer)
    print('LR Scheduler :', scheduler.__class__.__name__)



    # initialize trainer
    trainer = Trainer(model=model, optimizer=optimizer, batch_size=batch_size,
                      print_every=512, scheduler=scheduler,device=device, beta=0.01)
    trainer.run_name = "loglr"+"{:.0f}".format(-1*np.log10(lr))+"_"+"batch{:.0f}".format(batch_size)

    early_stopper = EarlyStopping(patience=10, min_delta=.01, verbose=True)
    trainer.train(train_loader, test_loader, early_stopper, epochs=50, save=True, early_stop=False)


if __name__ == '__main__':
    train_main(lr=1.e-3, batch_size=64)
