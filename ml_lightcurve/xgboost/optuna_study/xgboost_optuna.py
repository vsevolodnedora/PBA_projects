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
from optuna.trial import TrialState
from optuna.integration import XGBoostPruningCallback
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, Normalize
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

working_dir = os.getcwd()+"/"

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

    def _load_all_data(self):
        with h5py.File(self.working_dir+"X_train.h5", "r") as f:
            X = np.array(f["X"])
        with h5py.File(self.working_dir+"y_train.h5", "r") as f:
            y = np.array(f["y"])
        return (X, y)

    def get_normalized_train_data(self):
        X, y = self._load_all_data()
        X_norm = self.transform_x(X)
        y_norm = self.transform_y(y)
        return (X_norm, y_norm)

    def get_x_for_pars(self, pars:dict):
        pars["time"] = 0.
        pars = [pars[key] for key in self.features] # to preserve the order for inference
        _pars = np.vstack(([np.array(pars) for _ in range(len(self.times))]))
        _pars[:,-1] = self.times
        # print(_pars)
        return self.transform_x( _pars )

    def get_lc(self, pars):
        X, y = self._load_all_data()
        mask = np.ones_like(y, dtype=bool)
        x_ = self.get_x_for_pars(pars)
        # run for all features except time
        for i in range(len(self.features)-1):
            i_mask = X[:,i] == x_[0,i]
            mask = mask & i_mask
        # assert np.sum(mask) == len(self.times)
        lc = y[mask]
        return lc

class XGBoostObjetive():
    """
        Class for optuna_study training RF model
    """
    base_pars = {
        "verbosity": 0,  # 0 (silent) - 3 (debug)
        "objective": "reg:squarederror",
        "seed": 42,
        "n_jobs": 20, # number of parallel threads,
        "eval_metric":"rmse",
        "early_stopping_rounds":100,
    }
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_eval_model(self, params:dict, n_splits=10, n_repeats=1):
        model = xgb.XGBRegressor(**params)
        rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        )
        X_values = self.X
        y_values = self.y
        y_pred = np.zeros_like(y_values)
        for train_index, test_index in rkf.split(X_values):
            X_A, X_B = X_values[train_index, :], X_values[test_index, :]
            y_A, y_B = y_values[train_index], y_values[test_index]
            model.fit(
                X_A,
                y_A,
                eval_set=[(X_B, y_B)],
                verbose=0,
            )
            y_pred[test_index] += model.predict(X_B)
        y_pred /= n_repeats
        # pruning_callback = XGBoostPruningCallback(trial, 'valid-aft-nloglik')

        return (model, np.sqrt(mean_squared_error(self.y, y_pred)))

    def __call__(self, trial:optuna.trial.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5,log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6),
            "subsample": trial.suggest_float("subsample", 0.4, 0.8),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0,log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0,log=True),
            "gamma": trial.suggest_float("lambda", 1e-8, 10.0,log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000,log=True)
        }
        params.update(copy.deepcopy(self.base_pars))

        pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")

        params["callbacks"] = [pruning_callback]

        _, rmse = self.train_eval_model(params=params)

        return rmse
        # # Define hyperparameter search space
        # base_params = {
        #     'verbosity': 0,
        #     'objective': 'reg:squarederror',
        #     'eval_metric': 'rmse',
        #     'tree_method': 'hist' # Faster histogram optimized approximate greedy algorithm.
        # }  # Hyperparameters common to all trials
        # params = {
        #     'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        #     'max_depth': trial.suggest_int('max_depth', 6, 10), # Extremely prone to overfitting!
        #     'n_estimators': trial.suggest_int('n_estimators', 400, 4000, 400), # Extremely prone to overfitting!
        #     'eta': trial.suggest_float('eta', 0.007, 0.013), # Most important parameter.
        #     'subsample': trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1),
        #     'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.2, 0.9, 0.1),
        #     'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.2, 0.9, 0.1),
        #     'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 1e4), # I've had trouble with LB score until tuning this.
        #     'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4), # L2 regularization
        #     'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4), # L1 regularization
        #     'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),
        # }
        # params.update(base_params)
        # pruning_callback = optuna.integration.XGBoostPruningCallback(
        #     trial, f'valid-{base_params["eval_metric"]}'
        # )
        #
        # bst = xgb.train(params, self.dtrain, num_boost_round=10000,
        #                 evals=[(self.dtrain, 'train'),
        #                        (self.dvalid, 'valid')],
        #                 early_stopping_rounds=50,
        #                 verbose_eval=False,
        #                 callbacks=[pruning_callback])
        # if bst.best_iteration >= 25:
        #     return bst.best_score
        # else:
        #     return np.inf  # Reject models with < 25 trees

def save_study_results(working_dir:str, study:optuna.study.Study):

    # save the whole study
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
        file.write(f" Value: {trial.value}\n")
        file.write(f" Numer: {trial.number}\n")
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
            # mask = pd.Series(True, index=df.index)
            # for col, value in req_pars.items():
            #     i_mask = df[col] == value
            #     mask = mask & i_mask
            #
            # lc = np.log10( np.array(df["flux"][mask]) ).flatten()
            # x_ = data.get_x_for_pars(req_pars)
            # X, y = data._load_all_data()w
            lc = np.log10( data.get_lc(req_pars) )

            assert len(lc) == len(times), " size mismatch for lc"

            l11, = axes[0, i].plot(times/86400, lc,ls='-', color='gray', alpha=0.5, label=f"Original") # for second legend
            l21, = axes[0, i].plot(times/86400, lc,ls='-', color=cmap(norm(freq)), alpha=0.5, label=f"{freq/1e9:.1f} GHz")
            l2s.append(l21)

            # # get light curve from model
            # lc_nn = np.log10( inference(rf, req_pars, times) )
            lc_nn = model.predict( data.get_x_for_pars(req_pars) )
            lc_nn = data.inverse_transform_y( lc_nn )
            lc_nn = np.log10(lc_nn)
            # print("hi")

            l12, = axes[0, i].plot(times/86400, lc_nn,ls='--', color='gray', alpha=0.5, label=f"cVAE") # for second legend
            l22, = axes[0, i].plot(times/86400, lc_nn,ls='--', color=cmap(norm(freq)), alpha=0.5)


            # plot difference
            axes[1, i].plot(times/86400, lc-lc_nn, ls='-', color=cmap(norm(freq)), alpha=0.5)
            # del mask
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

def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx
def plot_violin(data:Data, model, figfpath):

    req_times = np.array([0.1, 1., 10., 100., 1000., 1e4]) * 86400.

    X, y = data._load_all_data()
    y = data.transform_y(y)

    times = data.times
    lcs = np.reshape(y,
                     newshape=(len(y)//len(times),
                               len(times)))
    yhat = model.predict(X)
    lcs_nn = np.reshape(yhat,
                        newshape=(len(yhat)//len(times),
                                  len(times)))
    allfreqs = X[:,data.features.index("freq")]
    allfreqs = np.reshape(allfreqs,
                          newshape=(len(allfreqs)//len(times),
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


if __name__ == '__main__':
    # laod data
    data = Data(working_dir=working_dir)

    # get data normalized according to the config files
    X_norm, y_norm = data.get_normalized_train_data()

    # define objective function (__call__()) to be optimized
    objective = XGBoostObjetive(X=X_norm, y=y_norm)

    # instantiate the study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="example-study",
        direction='minimize',
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=sampler
    )

    # run all trials
    study.optimize(objective,
                   n_trials=1000,
                   callbacks=[lambda study, trial: gc.collect()])

    # record results
    save_study_results(working_dir=working_dir, study=study)

    params = {}
    params.update(objective.base_pars)
    params.update(study.best_trial.params)

    # Re-run training with the best hyperparameter combination
    print('Re-running the best trial... params = {}'.format(params))
    model, rmse = objective.train_eval_model(params=params)

    print('Saving the best trial... rmse = {}'.format(rmse))
    fpath = working_dir + "final_model.pkl"
    model.save_model('best_model.json')

    # model = XGBRegressor()
    # model.load_model(working_dir+'best_model.json')
    # data = Data(working_dir=working_dir)

    # plot model performance
    plot_violin(data, model, working_dir+"violin.png")
    tasks = [
        {"pars":{"eps_e":0.001,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":1e0}, "all_freqs":True},
        {"pars":{"eps_e":0.01,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.01}, "all_freqs":True},
        {"pars":{"eps_e":0.1,"eps_b":0.01,"eps_t":1.,"p":2.2,"theta_obs":0.,"n_ism":0.001}, "all_freqs":True},
    ]
    plot_lcs(tasks, model, data, working_dir+"lcs.png")
