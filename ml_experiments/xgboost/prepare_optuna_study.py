"""
Prepare the data for an optuna_study study
No normalization.
Split data into 2D array for X_train.h5 and 1D array y_train.h5
"""

import json
import os

import joblib
import numpy as np
import h5py
import pandas as pd
from sklearn import preprocessing

# fpath_to_collated = os.getcwd() + '/' + "collated.csv"
fpath_to_collated = "/media/vsevolod/T7/work/prj_kn_afterglow/SFHoTim276_135_135_45km_150mstg_B0_FUKA/collated.csv"


def prep_data(working_dir, df:pd.DataFrame, features_names:list):
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


if __name__ == '__main__':

    assert os.path.isfile(fpath_to_collated), "Collated file not found"
    df = pd.read_csv(fpath_to_collated, index_col=0)
    print(f"File loaded: {fpath_to_collated} {print(df.info(memory_usage='deep'))}")

    target_key = "flux"
    features_names = [col for col in list(df.columns) if col not in [target_key]] # here time is included

    optuna_study_dir = os.getcwd() + '/' + 'optuna_study/'
    if not os.path.isdir(optuna_study_dir): os.mkdir(optuna_study_dir)

    # save data for study
    prep_data(working_dir=optuna_study_dir, df=df, features_names=features_names)