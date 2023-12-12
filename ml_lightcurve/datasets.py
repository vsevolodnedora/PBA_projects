import os.path
import sys
import numpy as np
import pandas as pd
import torch
import gzip
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing

AFGRUNDIR = "/media/vsevolod/T7/work/prj_kn_afterglow/"
sim = {}; sim["name"] = "SFHoTim276_13_14_0025_150mstg_B0_HLLC"
collated_file_path = AFGRUNDIR + sim["name"] + '/' + "collated.csv"

assert os.path.isfile(collated_file_path), "Collated file not found"
df = pd.read_csv(collated_file_path, index_col=0)
print(f"File loaded: {collated_file_path} {print(df.info(memory_usage='deep'))}")

