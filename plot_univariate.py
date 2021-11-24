from typing import List
from functools import partial
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm


def plot_feature(
    feat: str,
    y_col: str = None,  # janky, making kwarg for easier functools.partial...
    df: pd.DataFrame = None,
    is_binary_outcome: bool = None,
    coerce_cat_thresh: float = 0.001,
    graph_sample: int = 10_000,
    cat_freq_thresh: float = 0.01,
    do_scatter: bool = False,
    ylabel: str = None,
) -> None:
    try:
        this_df = df.copy().dropna(subset=[feat, y_col])

        is_cat = (this_df[feat].dtype == "object") or (hasattr(this_df[feat], "cat"))

        cardinality = this_df[feat].nunique()
        rel_cardinality = cardinality / len(this_df)

        is_dt = "datetime" in this_df[feat].dtype.name  # HACK
        is_numeric = is_numeric_dtype(this_df[feat])
        is_binary = cardinality == 2
        plot_lowess = not is_binary_outcome

        graph_as_cat = (
            (is_cat or (rel_cardinality < coerce_cat_thresh))
            and not is_dt
            and not is_numeric
        ) or is_binary

        if graph_as_cat:
            freqs = this_df[feat].value_counts(normalize=True)
            # Filter on level of var occurred > cat_freq_thresh % of times; sort by freq
            freqs = freqs.loc[freqs >= cat_freq_thresh]
            levels_to_eval = freqs.index
            if not list(levels_to_eval):
                return None  # very high cardinality, skip

            plt.figure(figsize=(12, 8))
            sns.catplot(
                x=feat,
                y=y_col,
                data=this_df,
                kind="point",
                join=False,
                order=levels_to_eval,
            )
            plt.xticks(rotation=90)
            plt.title(f"{feat} -> {y_col}?")
            if ylabel:
                plt.ylabel(ylabel)
            plt.show()

        # consider dt to be days since minimum TS
        elif is_dt and not is_numeric:

            min_ts = this_df[feat].min()
            days_since_min = (
                pd.to_datetime(this_df[feat]) - pd.to_datetime(this_df[feat]).min()
            ) / pd.to_timedelta("1d")

            empirical1pct, empirical99pct = (
                days_since_min.quantile(0.01),
                days_since_min.quantile(0.99),
            )
            fil_outliers = (days_since_min >= empirical1pct) & (
                days_since_min <= empirical99pct
            )
            graph_sample = min(graph_sample, len(this_df.loc[fil_outliers]))

            sns.regplot(
                x=days_since_min.sample(graph_sample, random_state=42),
                y=this_df[y_col].sample(graph_sample, random_state=42),
                scatter_kws={"alpha": 0.2},
                lowess=plot_lowess,
                logistic=is_binary_outcome,
                scatter=do_scatter,
            )
            plt.title(f"{feat} (as days since min.) ->  {y_col}?")
            if ylabel:
                plt.ylabel(ylabel)
            plt.show()

        # numeric feature, use regplot
        elif is_numeric:
            # confirm it can be cast to float
            _ = this_df[feat].astype("float")

            empirical1pct, empirical99pct = (
                this_df[feat].quantile(0.01),
                this_df[feat].quantile(0.99),
            )
            fil_outliers = (this_df[feat] >= empirical1pct) & (
                this_df[feat] <= empirical99pct
            )

            graph_sample = min(graph_sample, len(this_df.loc[fil_outliers]))

            sampled = (
                this_df.loc[fil_outliers, [feat, y_col]]
                .sample(graph_sample, random_state=42)
                .astype("float")
            )
            sns.regplot(
                x=feat,
                y=y_col,
                data=sampled,
                scatter_kws={"alpha": 0.2},
                lowess=plot_lowess,
                logistic=is_binary_outcome,
                scatter=do_scatter,
            )
            plt.title(f"{feat} -> {y_col}?")
            if ylabel:
                plt.ylabel(ylabel)
            plt.show()

        else:
            warnings.warn(f"Unhandled column {feat}")

    except Exception as err:
        warnings.warn(f"Error for feature {feat}.")
        warnings.warn(str(err))
        raise (err)
        pass


def plot_univariate(
    df: pd.DataFrame,
    feats: List[str],
    y_col: str,
    coerce_cat_thresh: float = 0.001,
    graph_sample: int = 10_000,
    cat_freq_thresh: float = 0.01,
    do_scatter: bool = False,
    ylabel: str = None,
) -> None:
    """
    Plot a list of features compared to outcome.
    df: pd.DataFrame, containing relevant data
    feats: list[str], colnames of x features to graph against your outcome
    y_col: str, name of your outcome column, assume it's continuous
    coerce_cat_thresh: float, will manually consider x cols to be cats if
      len(df.col.unique()) < cat_thresh * len(df.col)
      E.G. by default if less than 0.1% unique values, consider it to be categorical.
    graph_sample: int, how many data points to use for scatter graphs (gets messy with too many)
    cat_freq_thresh: float, % of the non-NA values of the column must be x in order to graph it.
      i.e. ignore very rare cats.
    return: None, will display graphs
    """

    is_binary_outcome = df[y_col].nunique() == 2

    plot_with_params = partial(
        plot_feature,
        y_col=y_col,
        df=df,
        is_binary_outcome=is_binary_outcome,
        coerce_cat_thresh=coerce_cat_thresh,
        graph_sample=graph_sample,
        cat_freq_thresh=cat_freq_thresh,
        do_scatter=do_scatter,
        ylabel=ylabel,
    )

    [plot_with_params(feat) for feat in feats]
