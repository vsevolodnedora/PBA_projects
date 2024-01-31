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
import gc,os,h5py,json,datetime,numpy as np
import pandas as pd
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
    def __init__(self, working_dir, batch_size, device:torch.device, lc_transform_method="minmax"):

        self.batch_size = batch_size
        self.working_dir = working_dir
        self.device = device
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

def loss_cvae(x, xhat, mu, logvar, beta):
    # bce = F.binary_cross_entropy(xhat, x, reduction='mean')
    mse = F.mse_loss(xhat, x, reduction='mean')
    # wmse = torch.sum((xhat - x) ** 2 / x ** 2) / (x.shape[0] * x.shape[1])
    kld_l = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp()) / x.shape[0] #/ 1e4
    # kld_o = -1. * F.kl_div(xhat, x, reduction='mean')
    loss = mse + beta * kld_l #+ 1 * kld_o
    return loss


def run(pars:dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("Running on GPU")
    else:
        print("Running on CPU")

    working_dir = os.getcwd() + '/'
    checkpoint_dir = None
    final_model_dir = None

    batch_size = pars["batch_size"] # 64
    dataset = LightCurveDataset(working_dir=working_dir, batch_size=batch_size,
                                device=device, lc_transform_method="minmax")

    # =================== MODEL ===================
    if pars["model_name"]==Kamile_CVAE.name:
        model = Kamile_CVAE(
            image_size=len(dataset.lcs[0]),  #  150
            hidden_dim=700,
            z_dim=10,#3 * len(dataset.features_names),
            c=len(dataset.features_names)
        )

    model = Kamile_CVAE(
        image_size=len(dataset.lcs[0]),  #  150
        hidden_dim=700,
        z_dim=10,#3 * len(dataset.features_names),
        c=len(dataset.features_names)
    )
    model.init_weights()
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Num of trainable params: {n_train_params}')
    if torch.cuda.device_count() > 1 and True:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        model.to(device)

    # =================== DATA ===================
    train_loader, test_loader = dataset.get_dataloader(test_split=.2)

    # =================== OPTIMIZER ===================
    lr = 1.e-3
    momentum = 0.9
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

    # =================== Scheduler ===================
    scheduler = select_scheduler(optimizer, lr_sch='step')

    # =================== STOPPER ===================
    early_stopper = EarlyStopping(patience=10, min_delta=.01, verbose=True)

    # =================== TRAIN & TEST ===================
    epochs = 50
    beta = 0.01 # or 'step'
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
        total_loss = 0.
        for i, (data, label, data_phys, label_phys) in enumerate(train_loader):
            num_steps += 1
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad() # Resets the gradients of all optimized tensors
            xhat, mu, logvar, z = model(data, label) # Forward pass only to get logits/output (evaluate model)
            loss = loss_cvae(data, xhat, mu, logvar, beta) # compute/store loss
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True.
            optimizer.step() # perform a single optimization step

            total_loss += loss.item()
        train_loss['Loss'].append(total_loss / len(train_loader) * dataset.batch_size)
        print("\t Train loss : %3.4f" % (train_loss['Loss'][-1]))

        # ------------- Validate -------------
        model.eval()
        total_loss = 0.
        with torch.no_grad():
            for i, (data, label, data_phys, label_phys) in enumerate(test_loader):
                data = data.to(device)
                label = label.to(device)
                xhat, mu, logvar, z = model(data, label) # evaluate model on the data
                loss = loss_cvae(data, xhat, mu, logvar, beta) #  computes dloss/dx requires_grad=False

                total_loss += loss.item()
        valid_loss['Loss'].append(total_loss / len(test_loader) * dataset.batch_size)
        print("\t Test loss : %3.4f" % (valid_loss['Loss'][-1]))

        # ------------- Update -------------
        if scheduler is not None:
            if 'ReduceLROnPlateau' == scheduler.__class__.__name__:
                scheduler.step(valid_loss['Loss'][-1])
            else:
                scheduler.step(epoch)

        epoch_time = datetime.datetime.now() - e_time
        elap_time = datetime.datetime.now() - time_start
        print(f"\t Time={elap_time.seconds/60:.2f} m  Time/Epoch={epoch_time.seconds:.2f} s ")

        # ------------- Save chpt -------------
        if not checkpoint_dir is None:
            fname = '%s_%d.chkpt' % (checkpoint_dir, epoch)
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
        if (early_stopper):
            if early_stopper.early_stop:
                print("\t Early stopping")
                break

    # =================== SAVE ===================

    if not final_model_dir is None:
        fname = final_model_dir + "model.pt"
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
        out_train = pd.DataFrame.from_dict(train_loss)
        out_train.to_csv(fname.replace('.pt','_train_loss.csv'), index=False)
        out_test = pd.DataFrame.from_dict(valid_loss)
        out_test.to_csv(fname.replace('.pt','_valid_loss.csv'), index=False)

if __name__ == '__main__':
    pars = {
        "batch_size":64
    }
    run(pars)

