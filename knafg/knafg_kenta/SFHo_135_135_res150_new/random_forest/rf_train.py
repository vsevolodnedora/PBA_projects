import pandas as pd
import os.path
import os
import h5py
import copy
import numpy as np
import tqdm
import itertools
import joblib
import gc
import optuna
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, Normalize
import json

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold

collated_file_path = os.getcwd() + '/' + "collated.csv"
assert os.path.isfile(collated_file_path), "Collated file not found"
df = pd.read_csv(collated_file_path, index_col=0)
print(f"File loaded: {collated_file_path} {print(df.info(memory_usage='deep'))}")

target = "flux"
features_names = [col for col in list(df.columns) if col not in [target]] # here time is included

working_dir = os.getcwd() + '/' + 'rf_optuna/'
if not os.path.isdir(working_dir): os.mkdir(working_dir)

# ---- EXTRACT DATA FROM COLLECTION ----

def prep_data(df:pd.DataFrame, features_names:list):
    # remove unnecessary features
    print(f'Extraction times: {df["text"].unique()}')
    df = df.loc[df["text"] == 32]
    df.drop(["text"], axis=1, inplace=True)
    features_names.remove("text")

    # extract X and y as arrays
    X = df.copy()
    y = np.array( X.loc[:,"flux"] )
    X = np.array( X.loc[:,features_names] )
    print(f"X shape: {X.shape}, y shape={y.shape}")

    # save metadata of the original file
    with h5py.File(working_dir+"train_data_meta.h5", "w") as f:
        f.create_dataset(name="times", data=np.array( df["time"].unique()))
        f.create_dataset(name="freqs", data=np.array( df["freq"].unique()))
        f.create_dataset(name="features", data=features_names)

        # save metadata of the original file
    with h5py.File(working_dir+"X_train.h5", "w") as f:
        f.create_dataset(name="X", data=X)
    with h5py.File(working_dir+"y_train.h5", "w") as f:
        f.create_dataset(name="y", data=y)

    # normalize X
    scaler_x = preprocessing.MinMaxScaler()
    scaler_x.fit(X)
    # norm_X = scaler_x.transform(X)
    fname_x = working_dir+'x_scaler.pkl'
    joblib.dump(scaler_x, fname_x)
    # scaler_x = joblib.load(working_dir+'x_scaler.pkl')

    # normalize y
    # y = np.log10(y)

    # save info for future
    with open(working_dir+"train_data_info.json", 'w') as outfile:
        json.dump(
            {
                "target": "flux",
                "x_scaler": "none",
                "y_scaler": "log10",
                "features": features_names
            },
            outfile)
prep_data(df,features_names)

# ---- CREATE DATA METADATA ----

class Data():
    def __init__(self, working_dir):
        self.working_dir = working_dir
        # load training data information
        with open(self.working_dir+"train_data_info.json", 'r') as infile:
            self.info = json.load(infile)
        self.features = self.info['features']

        # create xscaler
        if (self.info["x_scaler"].__contains__(".pkl")):
            scaler = joblib.load(self.working_dir+'x_scaler.pkl')
            self.transform_x = lambda X: scaler.transform(X)
            self.inverse_transform_x = lambda X_norm: scaler.inverse_transform(X_norm)
        elif (self.info["x_scaler"] == "none"):
            self.transform_x = lambda X: X
            self.inverse_transform_x = lambda X_norm: X_norm
        else: raise NotImplementedError("Not implemented yet")

        # read yscaler
        if (self.info["y_scaler"] == "log10"):
            self.transform_y = lambda y: np.log10(y)
            self.inverse_transform_y = lambda y_norm: np.power(10., y_norm)
        else: raise NotImplementedError("Not implemented yet")

        # load train data meta
        with h5py.File(self.working_dir+"train_data_meta.h5", "r") as f:
            self.times = np.array(f["times"])
            self.freqs = np.array(f["freqs"])
        print(f"times={self.times.shape} freq={self.freqs.shape}")

    def _load_train_data(self):
        with h5py.File(self.working_dir+"X_train.h5", "r") as f:
            X = np.array(f["X"])
        with h5py.File(self.working_dir+"y_train.h5", "r") as f:
            y = np.array(f["y"])
        return (X, y)

    def get_normalized_train_data(self):
        X, y = self._load_train_data()
        X_norm = self.transform_x(X)
        y_norm = self.transform_y(y)
        return (X_norm, y_norm)

    def get_x_for_pars(self, pars:dict):
        pars["time"] = 0.
        pars = [pars[key] for key in self.features] # to preserve the order for inference
        _pars = np.vstack(([np.array(pars) for _ in range(len(self.times))]))
        _pars[:,-1] = self.times
        print(_pars)
        return self.transform_x( _pars )
data = Data(working_dir)
X_norm, y_norm = data.get_normalized_train_data()

# ---- USE OPTUNA TO STUDY
class RFObjective():
    """
        Class for optuna_study training RF model
    """

    def __init__(self, X, y):
        self.rf_int_pars = {
            "n_estimators":[5, 50],
            "max_depth":[2, 20],
            "min_samples_split":[1, 10],
            "min_samples_leaf":[1, 10]
        }
        self.X = X
        self.y = y
        # self.data = data

    def evaluate(self, model, test_features, test_labels):

        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape

        print('\tModel Performance:')
        print('\tAverage Error: {:0.2e}.'.format(np.mean(errors)))
        print('\tAccuracy = {:0.2f}%.'.format(accuracy))

    def __call__(self, trial:optuna.trial.Trial):
        # Define the hyperparameters to be tuned
        pars = {}
        for par, range in self.rf_int_pars.items():
            pars[par] = trial.suggest_int(par, range[0], range[1])

        # random forest regressor with suggested hyperparameters
        rf = RandomForestRegressor(
            **pars,
            random_state=23,
            oob_score=False,
            n_jobs=8
        )

        # K-Fold Cross-Validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform cross-validation and return the average score
        scores = cross_val_score(rf, self.X, self.y,
                                 cv=kfold,
                                 scoring='neg_mean_squared_error',
                                 n_jobs=4)

        print(f"\t Trial: {np.mean(scores)} \t Pars:{pars}")
        # self.evaluate(rf, self.X, self.y)

        return np.mean(scores)

study = optuna.create_study(
    study_name="example-study",
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)

study.optimize(RFObjective(X_norm, y_norm),
               n_trials=100,
               callbacks=[lambda study, trial: gc.collect()])
print(f"Best trial: {study.best_trial.params}")

fpath = working_dir + "optuna_rf_study.pkl"
joblib.dump(study, fpath)

# ------- FIT BEST MODEL --------

# train model on the best parameters and entire dataset
rf = RandomForestRegressor(
    **study.best_trial.params,
    random_state=23,
    oob_score=False,
    n_jobs=8
)
print("Training model...")
rf = rf.fit(X_norm, y_norm)

fpath = working_dir + "best_rf.joblib"
print(f"Saving model {fpath}")
joblib.dump(rf, fpath) # compress=0

# -------- ASSESS PERFORMANCE --------

def plot_lcs(tasks, model, data:Data, figfpath):
    times = data.times
    freqs = data.freqs
    norm = LogNorm(vmin=np.min(freqs),
                   vmax=np.max(freqs))
    cmap_name = "coolwarm_r" # 'coolwarm_r'
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(2,3, figsize=(15,5), sharex='all',
                             gridspec_kw={'height_ratios': [3, 1]})

    for i, task in enumerate(tasks):
        # pars = [req_pars[feat] for feat in features_names]
        l2s = []
        for freq in freqs:
            req_pars = copy.deepcopy(task["pars"])
            req_pars["freq"] = freq

            # Create a boolean mask based on the dictionary
            mask = pd.Series(True, index=df.index)
            for col, value in req_pars.items():
                i_mask = df[col] == value
                mask = mask & i_mask

            lc = np.log10( np.array(df["flux"][mask]) ).flatten()

            assert len(lc) == len(times), " size mismatch for lc"

            l11, = axes[0, i].plot(times/86400, lc,ls='-', color='gray', alpha=0.5, label=f"Original") # for second legend
            l21, = axes[0, i].plot(times/86400, lc,ls='-', color=cmap(norm(freq)), alpha=0.5, label=f"{freq/1e9:.1f} GHz")
            l2s.append(l21)

            # # get light curve from model
            # lc_nn = np.log10( inference(rf, req_pars, times) )
            lc_nn = model.predict( data.get_x_for_pars(req_pars) )
            lc_nn = data.inverse_transform_y( lc_nn )
            lc_nn = np.log10(lc_nn)
            print("hi")

            l12, = axes[0, i].plot(times/86400, lc_nn,ls='--', color='gray', alpha=0.5, label=f"cVAE") # for second legend
            l22, = axes[0, i].plot(times/86400, lc_nn,ls='--', color=cmap(norm(freq)), alpha=0.5)


            # plot difference
            axes[1, i].plot(times/86400, lc-lc_nn, ls='-', color=cmap(norm(freq)), alpha=0.5)
            del mask
        axes[0, i].set_xscale("log")
        axes[1, i].set_xlabel(r"times [day]")

    axes[0,0].set_ylabel(r'$\log_{10}(F_{\nu})$')
    axes[1,0].set_ylabel(r'$\Delta\log_{10}(F_{\nu})$')

    first_legend = axes[0,0].legend(handles=l2s, loc='upper right')

    axes[0,0].add_artist(first_legend)
    axes[0,0].legend(handles=[l11,l12], loc='lower right')

    plt.tight_layout()
    plt.savefig(figfpath,dpi=256)
    plt.show()
tasks = [
    {"pars":{"eps_e":0.001,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":1e0}, "all_freqs":True},
    {"pars":{"eps_e":0.01,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.01}, "all_freqs":True},
    {"pars":{"eps_e":0.1,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.001}, "all_freqs":True},
]
plot_lcs(tasks, rf, data, working_dir+"lcs.png")

def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx
def plot_violin(data:Data, model, figfpath):

    req_times = np.array([0.1, 1., 10., 100., 1000., 1e4]) * 86400.

    times = data.times
    lcs = np.reshape(y_norm,
                     newshape=(len(y_norm)//len(times),
                               len(times)))
    yhat = model.predict(X_norm)
    lcs_nn = np.reshape(yhat,
                        newshape=(len(yhat)//len(times),
                                  len(times)))
    allfreqs = np.reshape(df["freq"],
                          newshape=(len(df["freq"])//len(times),
                                    len(times)))


    cmap_name = "coolwarm_r" # 'coolwarm_r'
    cmap = plt.get_cmap(cmap_name)

    # freqs = np.array(df["freq"].unique())
    # times = np.array(df["time"].unique())
    norm = LogNorm(vmin=np.min(data.freqs),
                   vmax=np.max(data.freqs))

    # log_lcs = np.log10(lcs)
    # log_nn_lcs = np.log10(lcs_nn)
    delta = lcs - lcs_nn
    print(delta.shape)

    fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex="all", sharey="all")
    ax = ax.flatten()
    for ifreq, freq in enumerate(data.freqs):
        i_mask1 = (allfreqs == freq).astype(int)#X[]#(np.array(df["freq"]) == freq).astype(bool)

        _delta = delta * i_mask1

        time_indeces = [find_nearest_index(times, t) for t in req_times]
        _delta = _delta[:, time_indeces]


        color = cmap(norm(data.freqs[0]))

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

        ax[ifreq].set_ylim(-0.5,0.5)


    # Create the new axis for marginal X and Y labels
    ax = fig.add_subplot(111, frameon=False)

    # Disable ticks. using ax.tick_params() works as well
    ax.set_xticks([])
    ax.set_yticks([])

    # Set X and Y label. Add labelpad so that the text does not overlap the ticks
    ax.set_xlabel(r"Time [days]", labelpad=20, fontsize=12)
    ax.set_ylabel(r"$\Delta \log_{10}(F_{\nu})$", labelpad=40, fontsize=12)

    plt.tight_layout()
    plt.savefig(figfpath,dpi=256)
    plt.show()
plot_violin(data, rf, working_dir+"violin.png")