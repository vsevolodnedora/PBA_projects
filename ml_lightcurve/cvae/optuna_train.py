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
import copy
import gc,os,h5py,json,datetime,numpy as np
import pandas as pd

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

class LightCurveDataset(Dataset):
    """
    LightCurve dataset
    Dispatches a lightcurve to the appropriate index
    """
    def __init__(self, working_dir, batch_size, lc_transform_method="minmax"):

        self.batch_size = batch_size
        self.working_dir = working_dir
        self.lc_transform_method = lc_transform_method

        # load dataset
        self.load_preprocessed_dataset()

        # scale parameters
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.pars)
        self.pars_normed = self.scaler.transform(self.pars)
        # inverse transform
        # inverse = scaler.inverse_transform(normalized)
        if np.min(self.pars_normed) < 0. or np.max(self.pars_normed) > 1.01:
            raise ValueError(f"Parameter normalization error: min={np.min(self.pars_normed)} max={np.max(self.pars_normed)}")

        # preprocess lcs
        self.lcs_log_norm = self._transform_lcs(self.lcs)
        if np.min(self.lcs_log_norm) < 0. or np.max(self.lcs_log_norm) > 1.01:
            raise ValueError(f"LC normalization error: min={np.min(self.lcs_log_norm)} max={np.max(self.lcs_log_norm)}")

    def load_preprocessed_dataset(self):
        # Load Prepared data
        with h5py.File(self.working_dir+"X.h5","r") as f:
            self.lcs = np.array(f["X"])
            self.times = np.array(f["time"])
        with h5py.File(self.working_dir+"Y.h5","r") as f:
            self.pars = np.array(f["Y"])
            self.features_names = list([str(val.decode("utf-8")) for val in f["keys"]])

        print(f"lcs={self.lcs.shape}, pars={self.pars.shape}, times={self.times.shape}")
        print(f"lcs={self.lcs.min()}, {self.lcs.max()}, pars={self.pars.min()} "
              f"{self.pars.max()}, times={self.times.shape}")
        print(self.features_names)
        assert self.pars.shape[0] == self.lcs.shape[0], "size mismatch between lcs and pars"
        self.len = len(self.lcs)

    # return (pars, lcs, times)

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

    def get_dataloader(self, test_split=0.2):
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

        train_loader = DataLoader(self, batch_size=self.batch_size,
                                  sampler=train_sampler, drop_last=False)
        test_loader = DataLoader(self, batch_size=self.batch_size,
                                 sampler=test_sampler, drop_last=False)

        return (train_loader, test_loader)

class EarlyStopping:
    """Early stops the training if validation loss doesn't
        improve after a given patience."""
    def __init__(self, pars):
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

def select_model(device, pars):

    name = pars["name"]
    del pars["name"]
    if name == Kamile_CVAE.name:
        model = Kamile_CVAE(**pars)
    else:
        raise NameError(f"Model {name} is not recognized")

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Num of trainable params: {n_train_params}')

    if torch.cuda.device_count() > 1 and True:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # model.to(device)
    model=torch.compile(model, backend="aot_eager").to(device)
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
            print(f'\tReset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def run(pars:dict,device,model,loss_cvae,optimizer,scheduler,early_stopper):

    # ==================== DATA ====================

    train_loader, test_loader = dataset.get_dataloader(test_split=.2)

    # =================== TRAIN & TEST ===================

    epochs = pars["epochs"]
    beta = pars["beta"] # or 'step'
    time_start = datetime.datetime.now()
    train_loss = {key: [] for key in ["KL_latent", "BCE", "MSE", "Loss"]}
    valid_loss  = {key: [] for key in ["KL_latent", "BCE", "MSE", "Loss"]}
    epoch, num_steps = 0, 0
    for epoch in range(epochs):
        e_time = datetime.datetime.now()
        print(f"----------------| Epoch {epoch}/{epochs} |-----------------")

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
        # train_loss['Loss'].append(losses / len(train_loader) * dataset.batch_size)
        train_loss['Loss'].append(np.mean(losses))
        print("\t Train loss : %.3e" % (train_loss['Loss'][-1]))

        # ------------- Validate -------------
        model.eval()
        losses = []
        with torch.no_grad():
            for i, (data, label, data_phys, label_phys) in enumerate(test_loader):
                data = data.to(device)
                label = label.to(device)
                xhat, mu, logvar, z = model(data, label) # evaluate model on the data
                loss = loss_cvae(data, xhat, mu, logvar, beta) #  computes dloss/dx requires_grad=False

                losses.append(loss.item())
        # valid_loss['Loss'].append(losses / len(test_loader) * dataset.batch_size)
        valid_loss['Loss'].append(np.mean(losses))
        print("\t Test loss : %.3e" % (valid_loss['Loss'][-1]))

        # ------------- Update -------------
        if scheduler is not None:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(valid_loss['Loss'][-1])
            else:
                scheduler.step()

        epoch_time = datetime.datetime.now() - e_time
        elap_time = datetime.datetime.now() - time_start
        print(f"\t Time={elap_time.seconds/60:.2f} m  Time/Epoch={epoch_time.seconds:.2f} s ")

        # ------------- Save chpt -------------
        if not pars["checkpoint_dir"] is None:
            fname = '%s_%d.chkpt' % (pars["checkpoint_dir"], epoch)
            print("\t Saving checkpoint")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'beta': beta_scheduler(beta, epoch),
                'train_losses':valid_loss['Loss'][-1],
                'test_losses':valid_loss['Loss'][-1],
                'train_batch': len(train_loader),
                'test_batch':len(test_loader)
            }, fname)

        # ------------- Stop if -------------
        if (early_stopper(valid_loss['Loss'][-1])):
            break

    # =================== SAVE ===================

    if not pars["final_model_dir"] is None:
        fname = pars["final_model_dir"] + "model.pt"
        print(f"Saving model {fname}")
        torch.save({
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_loss,
            'valid_losses': valid_loss,
            'metadata':{
                "batch_size":dataset.batch_size,
                "beta":beta,
                "epochs":epochs,
                "last_epoch":epoch},
            "dataset":{
                "x_transform":"minmax",
                "y_transform":"log_minmax"
            }
        }, fname)

        print(f"Saving loss history")
        res = {"train_losses":train_loss["Loss"], "valid_losses":valid_loss["Loss"]}
        pd.DataFrame.from_dict(res).to_csv(fname.replace('.pt','_loss_history.csv'), index=False)

    # ================== CLEAR ===================

    model.apply(reset_weights)

    del train_loss
    del valid_loss
    gc.collect()

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

def plot_lcs(tasks, model, dataset, model_dir):

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
    plt.show()


def analyze(dataset,model,device,model_dir):
    state = torch.load(model_dir+"model.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)

    tasks = [
        {"pars":{"eps_e":0.001,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":1e0}, "all_freqs":True},
        {"pars":{"eps_e":0.01,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.01}, "all_freqs":True},
        {"pars":{"eps_e":0.1,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.001}, "all_freqs":True},
    ]
    plot_lcs(tasks, model, dataset, model_dir)

    return (model, state)

if __name__ == '__main__':

    working_dir = os.getcwd() + '/'
    checkpoint_dir = None#"new_run/"
    final_model_dir = "new_run/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("Running on GPU")
    else:
        print("Running on CPU")


    batch_size = 64
    dataset = LightCurveDataset(working_dir=working_dir, batch_size=batch_size,
                                lc_transform_method="minmax")

    # =================== INIT ===================
    model_pars = {"name":"Kamile_CVAE",
                  "image_size":len(dataset.lcs[0]),
                  "hidden_dim":150, # 700
                  "z_dim":4*len(dataset.features_names),
                  "c":len(dataset.features_names),
                  "init_weights":True
                  }
    model = select_model(device, copy.deepcopy(model_pars))
    loss_pars = {
        "mse_or_bce":"mse",
        "reduction":"mean", # sum
        "kld_norm":True,
        "use_beta":True
    }
    loss = Loss(loss_pars)
    optimizer_pars = {
        "name":"Adam",
        "lr":1.e-3
    }
    optimizer = select_optimizer(model, copy.deepcopy(optimizer_pars))
    scheduler_pars = {
        "name":"step",
        "step_size":10,
        "gamma":0.01
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
    early_stopper = EarlyStopping(copy.deepcopy(early_stopping_pars))

    # ================== RUN ===================
    main_pars = {
        "epochs":50,
        "beta":"step",
        "final_model_dir":"new_run/",
        "checkpoint_dir":None
    }
    run(main_pars,device,model,loss,optimizer,scheduler,early_stopper)

    # ================= ANALYZE ================

    analyze(dataset,model,device,final_model_dir)
