import numpy as np
import h5py
import pandas as pd
from glob import glob
import os
import re
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mticker
# settings
# color_pal = sns.color_palette()
from tqdm import tqdm

import os
import warnings
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

if __name__ == '__main__':
    pass