# Jan boilerplate, rev 2019/07/24

from datetime import datetime
import os
import sys
import pprint
import gc

import numpy as np
import pandas as pd

import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sklearn as skl

import pymc3 as pm

import xgboost as xgb

# This notebook expects Python 3.
assert sys.version_info >= (3, 4)

print(f"Python version: {sys.version}")
pd.show_versions()
print("Working directory: ", os.getcwd())
print(f"Current time {datetime.now()}")

# Display more rows, columns
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 140)

# Auto reload extensions before executing,
#   so we can edit references without restarting kernel.
#%load_ext autoreload
#%autoreload 2

# Print everything without requiring print statements.
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Wide window (fill up screen horizontally)
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))


# # # Graph styling

# big graphs
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

sns.set_context("talk")
sns.set_palette("GnBu")
sns.set_style("whitegrid")
# fix palette
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (10.6, 6)
# fonts
plt.rcParams["font.sans-serif"] = [
    "Univers LT Std 55 Roman",
    "Frutiger CE 55 Roman",
    "SchulbuchNord Normal",
    "Helvetica",
    "sans-serif",
]
plt.rcParams["font.serif"] = ["Optima", "serif"]
plt.rcParams["font.monospace"] = ["Roboto Mono"]
plt.rcParams["font.family"] = "sans-serif"
# big text
font = {"family": "sans-serif", "weight": "regular", "size": 16}
plt.rc("font", **font)
# slightly thinner linewidth, smaller markers
mpl.rcParams["lines.linewidth"] = 2.0
mpl.rcParams["lines.markersize"] = 6
# background
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
# plot colors
from cycler import cycler

hexes = [
    "#008fd5",
    "#d14e23",
    "#8ac42f",
    "#54904f",
    "#5cadab",
    "#540f91",
    "#ff1cb0",
    "#ffa01c",
]
colors = cycler("color", hexes)
plt.rcParams["axes.prop_cycle"] = colors

# # #


# default percentiles
percentiles = [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]

# pretty printer
pp = pprint.PrettyPrinter(indent=4)
p = lambda x: pp.pprint(x)

# Plotly initialization
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)


# # #
# Package versions for reproducibility
import pkg_resources
import types


def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package,
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to had
        # exceptions to this list manually!
        poorly_named_packages = {"PIL": "Pillow", "sklearn": "scikit-learn"}
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name


imports = list(set(get_imports()))
# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name != "pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))


# # #
# Convenience functions


def do_gc() -> None:
    """
    Prevent memory explosion from unclosed figures
    """
    plt.cla(), plt.clf(), plt.close("all")
    import gc

    gc.collect()
    return None


def auto_encode_categoricals(
    df: pd.DataFrame, thresh_uniq_vals: float = 0.01, verbose: bool = True
) -> pd.DataFrame:
    """
    Convert all object / string columns with (# uniq vals / # total vals) < threshold to category.

    Note: Rarely, a column will need to be 'object' (Python object) type 
      to be compatible with Pandas functions.

    df: pd.DataFrame, to downcast to categorical
    thresh_uniq_vals: float, proportion of unique values to require to downcast
    """
    cols = df.select_dtypes(include=["object"]).columns.tolist()
    thresh_absolute = thresh_uniq_vals * len(df)
    cat_cols = [col for col in cols if (len(df[col].unique()) < thresh_absolute)]
    if verbose:
        print("Converting these columns to categorical:")
        print(cat_cols)
    # Have to use a for here due to pd limitation:
    for col in cat_cols:
        df[col] = df[col].copy(deep=True).astype("category")
    return df


def downcast_int(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric types in a DataFrame to save memory.
    Note: this will end up dropping duplicated columns, 
          but one would hope you haven't duplicated columns.
    Note 2: some Pandas operations may only exist for floats or higher-bit ints.
    
    df: pd.DataFrame, to downcast
    """
    full = df.copy()
    # We can run into issues if we have duplicated columns - drop them.
    full = df.loc[:, ~df.columns.duplicated()]
    df = full.select_dtypes(include=[np.number])
    num_cols = df.columns
    non_num_cols = set(full.columns) - set(df.columns)
    for col in num_cols:
        not_null = not df[col].isnull().any()
        not_small = df[col].abs().mean() > 1.10  # avoid round small floats to 0
        not_fractional = np.isclose(df[col].values, df[col].round(0).values).all()
        downcast = not_null and not_small and not_fractional
        if downcast:
            if verbose:
                print(f"Downcast column {col}")
            df.loc[:, col] = pd.to_numeric(df[col].copy(deep=True), downcast="integer")
        else:
            pass

    df = pd.concat([full[list(non_num_cols)], df], axis="columns")
    return df


do_gc
