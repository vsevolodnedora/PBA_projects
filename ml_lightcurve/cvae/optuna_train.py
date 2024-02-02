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

rsync -arvP --append ./{optuna_train.py,model_cvae.py,X.h5,Y.h5} vnedora@urash.gw.physik.uni-potsdam.de:/home/enlil/vnedora/work/cvae_optuna/

"""

from typing import Dict, Any
import hashlib
import joblib

import copy
import gc,os,h5py,json,datetime,numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

import optuna
from optuna.trial import TrialState

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn import preprocessing
from model_cvae import Kamile_CVAE

# for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms

class LightCurveDataset(Dataset):
    """
    LightCurve dataset
    Dispatches a lightcurve to the appropriate index
    """
    def __init__(self):
        pass

    def load_normalize_data(self, data_dir, lc_transform_method="minmax", limit=None):
        self._load_preprocessed_dataset(data_dir,limit)
        self._normalize_data(lc_transform_method)

    def _load_preprocessed_dataset(self, data_dir, limit=None):
        # Load Prepared data
        with h5py.File(data_dir+"X.h5","r") as f:
            self.lcs = np.array(f["X"])
            self.times = np.array(f["time"])
        with h5py.File(data_dir+"Y.h5","r") as f:
            self.pars = np.array(f["Y"])
            self.features_names = list([str(val.decode("utf-8")) for val in f["keys"]])

        if not limit is None:
            print(f"LIMITING data to {limit}")
            self.lcs = self.lcs[:limit]
            self.pars = self.pars[:limit]

        print(f"lcs={self.lcs.shape}, pars={self.pars.shape}, times={self.times.shape}")
        print(f"lcs={self.lcs.min()}, {self.lcs.max()}, pars={self.pars.min()} "
              f"{self.pars.max()}, times={self.times.shape}")
        print(self.features_names)
        assert self.pars.shape[0] == self.lcs.shape[0], "size mismatch between lcs and pars"
        self.len = len(self.lcs)

    def _normalize_data(self, lc_transform_method="minmax"):
        # scale parameters
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.pars)
        self.pars_normed = self.scaler.transform(self.pars)
        # inverse transform
        # inverse = scaler.inverse_transform(normalized)
        if np.min(self.pars_normed) < 0. or np.max(self.pars_normed) > 1.01:
            raise ValueError(f"Parameter normalization error: min={np.min(self.pars_normed)} max={np.max(self.pars_normed)}")

        # preprocess lcs
        self.lc_transform_method = lc_transform_method
        self.lcs_log_norm = self._transform_lcs(self.lcs)
        if np.min(self.lcs_log_norm) < 0. or np.max(self.lcs_log_norm) > 1.01:
            raise ValueError(f"LC normalization error: min={np.min(self.lcs_log_norm)} max={np.max(self.lcs_log_norm)}")

    def __getitem__(self, index):
        """ returns image/lc, vars(params)[normalized], vars(params)[physical] """
        return (torch.from_numpy(self.lcs_log_norm[index]).to(torch.float), # .to(self.device)
                torch.from_numpy(self.pars_normed[index]).to(torch.float),  # .to(self.device) .reshape(-1,1)
                self.lcs[index],
                self.pars[index])


    def __len__(self):
        return len(self.lcs)

    def _transform_lcs(self, lcs):
        log_lcs = np.log10(lcs)

        self.lc_min = log_lcs.min()
        self.lc_max = log_lcs.max()

        if (self.lc_transform_method=="minmax"):
            #self.lcs_log_norm = (log_lcs - np.min(log_lcs)) / (np.max(log_lcs) - np.min(log_lcs))
            self.lc_scaler = preprocessing.MinMaxScaler(feature_range=(0.0001,0.9999)) # Otherwise max>1.000001

        elif (self.lc_transform_method=="standard"):
            self.lc_scaler = preprocessing.StandardScaler()

        self.lc_scaler.fit(log_lcs)
        return self.lc_scaler.transform(log_lcs)

    def inverse_transform_lc_log(self, lcs_log_normed):
        #return np.power(10, lcs_log_normed * (self.lc_max - self.lc_min) + self.lc_min)
        return np.power(10., self.lc_scaler.inverse_transform(lcs_log_normed))

    def _transform_pars(self, _pars):
        # print(_pars.shape, self.pars[:,0].shape)
        for i, par in enumerate(_pars.flatten()):
            if (par < self.pars[:,i].min()) or (par > self.pars[:,i].max()):
                raise ValueError(f"Parameter '{i}'={par} is outside of the training set limits "
                                 f"[{self.pars[:,i].min()}, {self.pars[:,i].max()}]")
        return self.scaler.transform(_pars)

    def _invert_transform_pars(self, _pars):
        self.scaler.inverse_transform(_pars)

    def get_dataloader(self, test_split=0.2, batch_size=32):
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
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(self, batch_size=batch_size,
                                  sampler=self.train_sampler, drop_last=False)
        test_loader = DataLoader(self, batch_size=batch_size,
                                 sampler=self.valid_sampler, drop_last=False)

        return (train_loader, test_loader)

class EarlyStopping:
    """Early stops the training if validation loss doesn't
        improve after a given patience."""
    def __init__(self, pars, verbose):
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
        self.verbose = verbose
        self.patience = pars["patience"]
        self.verbose = pars["verbose"]
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_delta = pars["min_delta"]

    def __call__(self, val_loss):

        current_loss = val_loss

        if self.best_score is None:
            self.best_score = current_loss
        elif abs(current_loss - self.best_score) < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = current_loss
            self.counter = 0
        return False

def select_optimizer(model, pars:dict)->torch.optim:
    if (pars["name"]=="Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=pars["lr"])
    elif (pars["name"]=="SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=pars["lr"],
                                    momentum=pars["momentum"], nesterov=True)
    else:
        raise NameError("Optimizer is not recognized")
    return optimizer

def select_scheduler(optimizer, pars:dict)->optim.lr_scheduler or None:
    """lr_sch='step'"""
    lr_sch = pars["name"]
    del pars["name"]
    if lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **pars)
    elif lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,**pars)
    elif lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,**pars)
    elif lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,**pars)
    else:
        scheduler = None
    return scheduler

def select_model(device, pars, verbose):

    name = pars["name"]
    del pars["name"]
    if name == Kamile_CVAE.name:
        model = Kamile_CVAE(**pars)
    else:
        raise NameError(f"Model {name} is not recognized")

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f'Num of trainable params: {n_train_params}')

    if torch.cuda.device_count() > 1 and True:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # model.to(device)
    model = torch.compile(model, backend="aot_eager").to(device)

    return model

def beta_scheduler(beta, epoch, beta0=0., step=50, gamma=0.1):
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
    if beta == 'step':
        return np.float32( beta0 + gamma * (epoch+1 // step) )
    else:
        return np.float32(beta)

class Loss:
    def __init__(self, pars):
        self.pars = pars
    def __call__(self, x, xhat, mu, logvar, beta):
        pars = self.pars
        if pars["mse_or_bce"]=="mse":   base = F.mse_loss(xhat, x, reduction=pars["reduction"])
        elif pars["mse_or_bce"]=="bce": base = F.binary_cross_entropy(xhat, x, reduction=pars["reduction"])
        else: base = 0

        kld_l = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
        if pars["kld_norm"]: kld_l = kld_l / x.shape[0]
        if pars["use_beta"]: kld_l *= beta

        loss = base + kld_l

        return loss


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()




def inference(pars:list, model:Kamile_CVAE, dataset:LightCurveDataset, device):
    # if len(pars) != model.z_dim:
    #     raise ValueError(f"Number of parameters = {len(pars)} does not match the model latent space size {model.z_dim}")
    # create state vector for intput data (repeat physical parameters for times needed)
    pars = np.asarray(pars).reshape(1, -1)
    # normalize parameters as in the training data
    normed_pars = dataset._transform_pars(_pars=pars)
    # generate prediction
    with torch.no_grad():
        # convert intput data to the format of the hidden space
        z = (torch.zeros((1, model.z_dim)).repeat((len(normed_pars), 1)).to(device).to(torch.float))
        # create the input for the decoder
        decoder_input = torch.cat((z, torch.from_numpy(normed_pars).to(device).to(torch.float)), dim=1)
        # perform reconstruction using model
        reconstructions = model.decoder(decoder_input)
    # move prediction to cpu and numpy
    reconstructions_np = reconstructions.double().cpu().detach().numpy()
    # undo normalization that was done in training data
    lc_nn = dataset.inverse_transform_lc_log(reconstructions_np)
    return lc_nn

def plot_lcs(tasks, model, device, dataset, model_dir):

    freqs = np.unique(dataset.pars[:,dataset.features_names.index("freq")])
    norm = LogNorm(vmin=np.min(freqs),
                   vmax=np.max(freqs))
    cmap_name = "coolwarm_r" # 'coolwarm_r'
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(2,3, figsize=(15,5), sharex='all',
                             gridspec_kw={'height_ratios': [3, 1]})
    for i, task in enumerate(tasks):
        req_pars = copy.deepcopy(task["pars"])
        # pars = [req_pars[feat] for feat in features_names]
        l2s = []
        for freq in freqs:

            # get light curve from training data
            mask = np.ones_like(dataset.pars[:,0],dtype=bool)
            req_pars["freq"] = freq
            for j, par in enumerate(dataset.features_names):
                i_mask = dataset.pars[:,j] == req_pars[par]
                if np.sum(i_mask) == 0:
                    raise ValueError(f"par={par} requested {req_pars[par]} "
                                     f"not found in data\n{np.unique(dataset.pars[:,j])}")
                mask = (mask & i_mask)
            if np.sum(mask) > 1:
                raise ValueError("error in extracting LC from train data")
            # expected array with one index
            lc = np.log10(np.array(dataset.lcs[mask, :])).flatten()



            l11, = axes[0, i].plot(dataset.times/86400, lc,
                                   ls='-', color='gray', alpha=0.5, label=f"Original") # for second legend
            l21, = axes[0, i].plot(dataset.times/86400, lc,
                                   ls='-', color=cmap(norm(freq)), alpha=0.5, label=f"{freq/1e9:.1f} GHz")
            l2s.append(l21)

            # get light curve from model
            # get list of parameters from dictionary
            req_pars_list = [np.float32(req_pars[feat]) for feat in dataset.features_names]
            lc_nn = np.log10(np.array(inference(req_pars_list, model, dataset, device)).flatten())
            l12, = axes[0, i].plot(dataset.times/86400, lc_nn,ls='--', color='gray', alpha=0.5, label=f"cVAE") # for second legend
            l22, = axes[0, i].plot(dataset.times/86400, lc_nn,ls='--', color=cmap(norm(freq)), alpha=0.5)


            # plot difference
            axes[1, i].plot(dataset.times/86400, lc-lc_nn, ls='-', color=cmap(norm(freq)), alpha=0.5)

        axes[0, i].set_xscale("log")
        axes[1, i].set_xlabel(r"times [day]")

    axes[0,0].set_ylabel(r'$\log_{10}(F_{\nu})$')
    axes[1,0].set_ylabel(r'$\Delta\log_{10}(F_{\nu})$')

    first_legend = axes[0,0].legend(handles=l2s, loc='upper right')

    axes[0,0].add_artist(first_legend)
    axes[0,0].legend(handles=[l11,l12], loc='lower right')

    plt.tight_layout()
    plt.savefig(model_dir+"/lcs.png",dpi=256)
    # plt.show()
    plt.close(fig)

def plot_violin(delta, dataset, model_dir):


    def find_nearest_index(array, value):
        ''' Finds index of the value in the array that is the closest to the provided one '''
        idx = (np.abs(array - value)).argmin()
        return idx

    cmap_name = "coolwarm_r" # 'coolwarm_r'
    cmap = plt.get_cmap(cmap_name)

    freqs = np.unique(dataset.pars[:,dataset.features_names.index("freq")])
    norm = LogNorm(vmin=np.min(freqs),
                   vmax=np.max(freqs))

    req_times = np.array([0.1, 1., 10., 100., 1000., 1e4]) * 86400.

    fig, axes = fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex="all", sharey="all")
    ax = ax.flatten()
    for ifreq, freq in enumerate(freqs):
        i_mask1 = dataset.pars[:, dataset.features_names.index("freq")] == freq

        _delta = delta[i_mask1]
        time_indeces = [find_nearest_index(dataset.times, t) for t in req_times]
        _delta = _delta[:, time_indeces]

        color = cmap(norm(freqs[0]))

        if np.sum(_delta) == 0:
            raise ValueError(f"np.sum(delta) == 0 delta={_delta.shape}")
        # print(_delta.shape)
        violin = ax[ifreq].violinplot(_delta, positions=range(len(req_times)),
                                      showextrema=False, showmedians=True)


        for pc in violin['bodies']:
            pc.set_facecolor(color)
        violin['cmedians'].set_color(color)
        for it, t in enumerate(req_times):
            ax[ifreq].vlines(it, np.quantile(_delta[:,it], 0.025), np.quantile(_delta[:,it], 0.975),
                             color=color, linestyle='-', alpha=.8)

        # ax[ifreq].hlines([-1,0,1], 0.1, 6.5, colors='gray', linestyles=['dashed', 'dotted', 'dashed'], alpha=0.5)


        ax[ifreq].set_xticks(np.arange(0, len(req_times)))
        # print(ax[ifreq].get_xticklabels(), ax[ifreq])
        _str = lambda t : '{:.1f}'.format(t/86400.) if t/86400. < 1 else '{:.0f}'.format(t/86400.)
        ax[ifreq].set_xticklabels([_str(t) for t in req_times])

        ax[ifreq].annotate(f"{freq/1e9:.1f} GHz", xy=(1, 1),xycoords='axes fraction',
                           fontsize=12, horizontalalignment='right', verticalalignment='bottom')


    # Create the new axis for marginal X and Y labels
    ax = fig.add_subplot(111, frameon=False)

    # Disable ticks. using ax.tick_params() works as well
    ax.set_xticks([])
    ax.set_yticks([])

    # Set X and Y label. Add labelpad so that the text does not overlap the ticks
    ax.set_xlabel(r"Time [days]", labelpad=20, fontsize=12)
    ax.set_ylabel(r"$\Delta \log_{10}(F_{\nu})$", labelpad=40, fontsize=12)
    plt.tight_layout()
    plt.savefig(model_dir+"violin.png",dpi=256)

    # plt.show()

def plot_loss(model, loss_df, model_dir):
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(5,3))
    ax.plot(range(len(loss_df)), loss_df["train_losses"],ls='-',color='blue',label='training loss')
    ax.plot(range(len(loss_df)), loss_df["valid_losses"],ls='-',color='red',label='validation loss')
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(model_dir+"/loss.png",dpi=256)
    # plt.show()
    plt.close(fig)

def analyze(dataset,model,device,model_dir,verbose):
    """ returns rms between log(light curves) and log(predicted light curves) """
    state = torch.load(model_dir+"model.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)

    # analyze loss
    loss_df = pd.read_csv(model_dir+"model_loss_history.csv")
    plot_loss(model, loss_df, model_dir)

    # plot selected light curves
    tasks = [
        {"pars":{"eps_e":0.001,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":1e0}, "all_freqs":True},
        {"pars":{"eps_e":0.01,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.01}, "all_freqs":True},
        {"pars":{"eps_e":0.1,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.001}, "all_freqs":True},
    ]
    plot_lcs(tasks, model, device, dataset, model_dir)

    # compute difference between all light curves and NN light curves
    nn_lcs = np.vstack((
        [inference(dataset.pars[j, :], model, dataset, device)
            for j in range(len(dataset.pars[:,0]))]
    ))
    print(f"lcs={dataset.lcs.shape} nn_lcs={nn_lcs.shape}")
    log_lcs = np.log10(dataset.lcs)
    log_nn_lcs = np.log10(nn_lcs)
    delta = (log_lcs - log_nn_lcs)

    plot_violin(delta, dataset, model_dir)

    # return total error
    rmse = root_mean_squared_error(log_lcs, log_nn_lcs)
    if verbose:
        print("Total RMSE: {:.2e}".format(rmse))
    return rmse

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def make_dir_for_run(working_dir, trial, dict_, verbose):
    final_model_dir = f"{working_dir}/trial_{trial}/"
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    else:
        raise FileExistsError(final_model_dir)
    if verbose:
        print("Saving pars in {}".format(final_model_dir))
    with open(final_model_dir+"pars.json", "w") as outfile:
        json.dump(dict_, outfile)
    return final_model_dir

def objective(trial:optuna.trial.Trial):
    # ------------------------------
    do = True
    verbose = False
    print(f"TRIAL {trial.number}")
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("Running on GPU")
    else:
        print("Running on CPU")

    # =================== INIT ===================
    main_pars = {
        "batch_size":trial.suggest_int("batch_size", low=16, high=128, step=8) if do else 32,
        "epochs":100,
        "beta":"step"
    }

    model_pars = {"name":"Kamile_CVAE",
                  "image_size":len(dataset.lcs[0]),
                  "hidden_dim": trial.suggest_int("hidden_dim", low=50, high=1000, step=10) if do else 200,
                  "z_dim": trial.suggest_int("z_dim", low=4, high=64, step=4) if do else len(dataset.features_names),
                  "c":len(dataset.features_names),
                  "init_weights":True
                  }
    model = select_model(device, copy.deepcopy(model_pars), verbose)
    loss_pars = {
        "mse_or_bce":trial.suggest_categorical(name="mse_or_bce",choices=["mse","bce"]) if do else "mse",
        "reduction":trial.suggest_categorical(name="reduction",choices=["sum","mean"]) if do else "sum",
        "kld_norm":trial.suggest_categorical(name="kld_norm",choices=[True,False]) if do else True,
        "use_beta":trial.suggest_categorical(name="use_beta",choices=[True,False]) if do else True,
    }
    loss_cvae = Loss(loss_pars)
    optimizer_pars = {
        "name":"Adam",
        "lr":trial.suggest_float(name="lr", low=1.e-6, high=1.e-2, log=True) if do else 1.e-2
    }
    optimizer = select_optimizer(model, copy.deepcopy(optimizer_pars))
    scheduler_pars = {
        "name":"step",
        "step_size":trial.suggest_int("step_size", low=6, high=20, step=2) if do else 5,
        "gamma":trial.suggest_float(name="gamma", low=1e-5, high=1e-1, log=True) if do else 0.1
        # "name":"exp", "gamma":0.985,
        # "name":"cos", "T_max":50, "eta_min":1e-5,
        # "name":"plateau", "mode":'min', "factor":.5,"verbose":True
    }
    scheduler = select_scheduler(optimizer, copy.deepcopy(scheduler_pars))
    early_stopping_pars = {
        "patience":5,
        "verbose":True,
        "min_delta":0.
    }
    early_stopper = EarlyStopping(copy.deepcopy(early_stopping_pars),verbose)

    # ================== RUN ===================

    final_model_dir = make_dir_for_run(working_dir, trial.number,{
        **main_pars,**model_pars,**loss_pars,
        **optimizer_pars,**scheduler_pars,**early_stopping_pars
    }, verbose)

    main_pars["final_model_dir"] = final_model_dir
    main_pars["checkpoint_dir"] = None # final_model_dir

    # ==================== DATA ====================

    train_loader, valid_loader = dataset.get_dataloader(
        test_split=.2,
        batch_size=main_pars["batch_size"]
    )

    # =================== TRAIN & TEST ===================

    epochs = main_pars["epochs"]
    beta = main_pars["beta"] # or 'step'
    time_start = datetime.datetime.now()
    train_loss = {key: [] for key in ["KL_latent", "BCE", "MSE", "Loss"]}
    valid_loss  = {key: [] for key in ["KL_latent", "BCE", "MSE", "Loss"]}
    valid_mse  = {key: [] for key in ["sum", "mean"]}
    epoch, num_steps = 0, 0
    for epoch in range(epochs):
        e_time = datetime.datetime.now()
        if verbose:
            print(f"-----| Epoch {epoch}/{epochs} | Train/Valid {len(train_loader)}/{len(valid_loader)} |-------")

        beta = beta_scheduler(beta, epoch) # Scheduler for beta value

        # ------------- Train -------------
        model.train() # set model into training mode
        losses = []
        for i, (data, label, data_phys, label_phys) in enumerate(train_loader):
            num_steps += 1
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad() # Resets the gradients of all optimized tensors
            xhat, mu, logvar, z = model(data, label) # Forward pass only to get logits/output (evaluate model)
            loss = loss_cvae(data, xhat, mu, logvar, beta) # compute/store loss
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True.
            optimizer.step() # perform a single optimization step
            losses.append(loss.item())
        train_loss['Loss'].append(np.sum(losses)/len(dataset.train_sampler))
        if verbose:
            print(f"\t Train loss: {train_loss['Loss'][-1]:.2e}")
        if (not np.isfinite(train_loss['Loss'][-1])):
            return 1e10 # large number
        # ------------- Validate -------------
        model.eval()
        losses, mse_mean, mse_sum = [], [], []
        with torch.no_grad():
            for i, (data, label, data_phys, label_phys) in enumerate(valid_loader):
                data = data.to(device)
                label = label.to(device)
                xhat, mu, logvar, z = model(data, label) # evaluate model on the data
                loss = loss_cvae(data, xhat, mu, logvar, beta) #  computes dloss/dx requires_grad=False
                mse_mean.append(np.sqrt(F.mse_loss(xhat, data, reduction="mean").item()))
                mse_sum.append(np.sqrt(F.mse_loss(xhat, data, reduction="sum").item()))
                losses.append(loss.item())

        # all_y = torch.from_numpy(dataset.lcs_log_norm).to(torch.float), # .to(self.device)
        # all_x = torch.from_numpy(dataset.pars_normed).to(torch.float)

        # xhat, _, _, _ = model(all_y, all_x) # evaluate model on the data
        # mse_sum_ = np.sqrt(F.mse_loss(xhat, all_x, reduction="sum").item())
        # mse_sum__ = np.sum(mse_sum)

        # valid_loss['Loss'].append(losses / len(valid_loader) * dataset.batch_size)
        # valid_mse["mean"].append(np.sum(mse_mean)/len(valid_loader))
        # valid_mse["sum"].append(np.sum(mse_sum)/len(valid_loader))
        valid_loss['Loss'].append(np.sum(losses) / len(dataset.valid_sampler))
        if (not np.isfinite(train_loss['Loss'][-1])):
            return 1e10 # large number
        # ----------- Evaluate ------------
        # num_samples = 10
        # all_y = torch.from_numpy(dataset.lcs_log_norm).to(torch.float).to(device)
        # all_x = torch.from_numpy(dataset.pars_normed).to(torch.float).to(device)
        # model.eval()
        # y_pred = torch.empty_like(all_y)
        # with torch.no_grad():
        #     for k in range(len(all_x)):
        #         multi_recon = torch.empty((num_samples, len(dataset.lcs[0])))
        #         for i in range(num_samples):
        #             z = torch.randn(1, model.z_dim).to(device).to(torch.float)
        #             x = all_x[k:k+1].to(torch.float)
        #             z1 = torch.cat((z, x), dim=1)
        #             recon = model.decoder(z1)
        #             multi_recon[i] = recon
        #         mean = torch.mean(multi_recon, axis=0)
        #         y_pred[k] = mean
        # errors = (torch.sum(
        #     torch.abs(dataset.inverse_transform_lc_log(all_y)
        #               - dataset.inverse_transform_lc_log(y_pred)), axis=1)
        #           /torch.sum( dataset.inverse_transform_lc_log(all_y),axis=1))
        # torch.save(y_pred, './'+ARGS.exp_name+'/test_predictions.pt')
        # torch.save(errors, './'+ARGS.exp_name+'/test_epsilons.pt')

        if verbose:
            print(f"\t Valid loss: {valid_loss['Loss'][-1]:.2e}")
                  # f"<RMSE> {valid_mse['mean'][-1]:.2e} sum(RMSE)={valid_mse['sum'][-1]:.2e}")

        # ------------- Update -------------
        if not (scheduler is None):
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(valid_loss['Loss'][-1])
            else:
                scheduler.step()

        epoch_time = datetime.datetime.now() - e_time
        elap_time = datetime.datetime.now() - time_start
        if verbose:
            print(f"\t Time={elap_time.seconds/60:.2f} m  Time/Epoch={epoch_time.seconds:.2f} s ")

        # ------------- Save chpt -------------
        if not main_pars["checkpoint_dir"] is None:
            fname = '%s_%d.chkpt' % (main_pars["checkpoint_dir"], epoch)
            if verbose: print("\t Saving checkpoint")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'beta': beta_scheduler(beta, epoch),
                'train_losses':train_loss['Loss'][-1],
                'valid_losses':valid_loss['Loss'][-1],
                'train_batch': len(train_loader),
                'test_batch':len(valid_loader)
            }, fname)

        # ------------- Stop if -------------
        if (early_stopper(valid_loss['Loss'][-1])):
            break

        # ------------- Prune -------------
        trial.report(valid_loss['Loss'][-1], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # =================== SAVE ===================

    if not main_pars["final_model_dir"] is None:
        fname = main_pars["final_model_dir"] + "model.pt"
        if verbose: print(f"Saving model {fname}")
        torch.save({
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_loss,
            'valid_losses': valid_loss,
            'metadata':{
                "batch_size":main_pars["batch_size"],
                "beta":beta,
                "epochs":epochs,
                "last_epoch":epoch},
            "dataset":{
                "x_transform":"minmax",
                "y_transform":"log_minmax"
            }
        }, fname)

        if verbose: print(f"Saving loss history")
        res = {"train_losses":train_loss["Loss"], "valid_losses":valid_loss["Loss"]}
        pd.DataFrame.from_dict(res).to_csv(fname.replace('.pt','_loss_history.csv'), index=False)

    # ================== CLEAR ==========s=========

    model.apply(reset_weights)

    # ================= ANALYZE ================

    rmse = analyze(dataset,model,device,final_model_dir,verbose)
    if (not np.isfinite(rmse)):
        return 1e10 # large number
    return rmse



if __name__ == '__main__':

    working_dir = os.getcwd() + '/'

    dataset = LightCurveDataset()
    dataset.load_normalize_data(working_dir, "minmax", None)

    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,
                   n_trials=1000,
                   callbacks=[lambda study, trial: gc.collect()])

    print("Study completed successfully. Saving study")
    fpath = working_dir + "study.pkl"
    joblib.dump(study, fpath)

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    with open(working_dir+"summary.txt", 'w') as file:
        # Display the study statistics
        file.write("\nStudy statistics: \n")
        file.write(f"  Number of finished trials: {len(study.trials)}\n")
        file.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        file.write(f"  Number of complete trials: {len(complete_trials)}\n")

        trial = study.best_trial
        file.write("\nBest trial:\n")
        file.write(f"  Value: {trial.value}\n")
        file.write(f"  Numer: {trial.number}\n")
        file.write("  Params: \n")
        for key, value in trial.params.items():
            file.write("    {}: {}\n".format(key, value))

        # Find the most important hyperparameters
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)
        # Display the most important hyperparameters
        file.write('\nMost important hyperparameters:\n')
        for key, value in most_important_parameters.items():
            file.write('  {}:{}{:.2f}%\n'.format(key, (15-len(key))*' ', value*100))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start',
                                        'datetime_complete',
                                        'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by loss):\n {}".format(df))


