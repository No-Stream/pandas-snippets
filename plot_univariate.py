from typing import List
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_univariate(
    df,
    feats: List[str],
    y_col: str,
    cat_thresh: float = 0.001,
    graph_sample: int = 10_000,
    cat_freq_thresh: float = 0.01,
) -> None:
    """
    Plot a list of features compared to outcome.

    df: pd.DataFrame, containing relevant data
    feats: list[str], colnames of x features to graph against your outcome
    y_col: str, name of your outcome column, assume it's continuous
    cat_thresh: float, will manually consider x cols to be cats len(df.col.unique()) < cat_thresh * len(df.col)
      E.G. by default if less than 0.1% unique values, consider it to be categorical.
    graph_sample: int, how many data points to use for scatter graphs (gets messy with too many)
    cat_freq_thresh: float, % of the non-NA values of the column must be x in order to graph it.
      i.e. ignore very rare cats.

    return: None, will display graphs
    """

    is_binary_outcome = df[y_col].nunique() == 2

    for feat in feats:

        this_df = df.copy().dropna(subset=[feat, y_col])

        is_cat = (this_df[feat].dtype == "object") or (
            hasattr(this_df[feat], 'cat')
        )

        cardinality = this_df[feat].nunique()
        rel_cardinality = cardinality / len(this_df)

        is_dt = "datetime" in this_df[feat].dtype.name  # HACK

        if (is_cat or (rel_cardinality < cat_thresh)) and not is_dt:
            freqs = this_df[feat].value_counts(normalize=True)
            # Filter on level of var occurred 100+ times; sort by freq
            freqs = freqs.loc[freqs >= cat_freq_thresh]
            levels_to_eval = freqs.index

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
            plt.title(f"Outcome {y_col} vs {feat}")
            plt.show()

        # consider dt to be integer seconds since UNIX epoch
        else:
            if is_dt:
                min_ts = this_df[feat].min()
                epoch_since_min = (
                    pd.to_datetime(this_df[feat]) - pd.to_datetime("1970-01-01")
                ) / pd.to_timedelta("1s")
                # transform to set min ts at 0 for readability
                min_epoch = epoch_since_min.min()
                epoch_since_min = epoch_since_min - min_epoch

                empirical1pct, empirical99pct = (
                    epoch_since_min.quantile(0.01),
                    epoch_since_min.quantile(0.99),
                )
                fil_outliers = (epoch_since_min >= empirical1pct) & (
                    epoch_since_min <= empirical99pct
                )

                sns.regplot(
                    x=epoch_since_min.sample(graph_sample, random_state=42),
                    y=this_df[y_col].sample(graph_sample, random_state=42),
                    scatter_kws={"alpha": 0.2},
                    lowess=True,
                )
                plt.title(f"{y_col} vs {feat} (as seconds since min.)")
                plt.show()

            # numeric feature, use regplot
            else:
                try:
                    # confirm it can be cast to float
                    _ = this_df[feat].astype("float")

                    empirical1pct, empirical99pct = (
                        this_df[feat].quantile(0.01),
                        this_df[feat].quantile(0.99),
                    )
                    fil_outliers = (this_df[feat] >= empirical1pct) & (
                        this_df[feat] <= empirical99pct
                    )

                    plot_lowess = not is_binary_outcome
                    sns.regplot(
                        x=feat,
                        y=y_col,
                        data=this_df.loc[fil_outliers].sample(
                            graph_sample, random_state=42
                        ),
                        scatter_kws={"alpha": 0.2},
                        lowess=plot_lowess,
                        logistic=is_binary_outcome,
                    )
                    plt.title(f"Regress {feat} on {y_col}.")
                    plt.show()
                except Exception as err:
                    warnings.warn(err)
                    pass
