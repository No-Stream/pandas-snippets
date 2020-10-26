from typing import List 

def plot_univariate(df, feats: List[str], y_col: str, cat_thresh: int = 10, graph_sample: int = 2000) -> None:
    """
    Plot a list of features compared to outcome.

    df: pd.DataFrame, containing relevant data
    feats: list[str], colnames of x features to graph against your outcome
    y_col: str, name of your outcome column, assume it's continuous
    cat_thresh: int, will manually consider x cols to be cats if less than this count of unique values
    graph_sample: int, how many data points to use for scatter graphs (gets messy with too many)
    
    return: None, will display graphs

    """
    for feat in feats:
       
        this_df = df.copy().dropna(subset=[feat, y_col])
       
        is_cat = this_df[feat].dtype == 'object'
        cardinality = this_df[feat].nunique()
        is_dt = 'datetime' in this_df[feat].dtype.name # HACK
       
        if (is_cat or (cardinality < cat_thresh)) and not is_dt:
            plt.figure(figsize=(12,8))
            sns.catplot(
                x=feat,
                y=y_col,
                data=this_df,
                kind='point',
                join=False,
            )
            plt.xticks(rotation=90)
            plt.title(f'{y_col} vs {feat}')
            plt.show()
       
       
        else:
            if is_dt:
                min_ts = this_df[feat].min()
                epoch_since_min = (
                    pd.to_datetime(this_df[feat]) - pd.to_datetime("1970-01-01")
                ) / pd.to_timedelta("1s")
                # transform to set min ts at 0 for readability
                min_epoch = epoch_since_min.min()
                epoch_since_min = epoch_since_min - min_epoch

                empirical1pct, empirical99pct = epoch_since_min.quantile(.01), epoch_since_min.quantile(.99)
                fil_outliers = ((epoch_since_min >= empirical1pct) & (epoch_since_min <= empirical99pct))

                sns.regplot(
                    x=epoch_since_min.sample(graph_sample, random_state=42),
                    y=this_df[y_col].sample(graph_sample, random_state=42),
                    scatter_kws={'alpha': .2},
                    lowess=True,
                )
                plt.title(f'{y_col} vs {feat} (as seconds since min.)')
                plt.show()

            # numeric feature, use regplot
            else:
                try:
                    # confirm it can be cast to float
                    _ = this_df[feat].astype('float')
                   
                    empirical1pct, empirical99pct = this_df[feat].quantile(.01), this_df[feat].quantile(.99)
                    fil_outliers = ((this_df[feat] >= empirical1pct) & (this_df[feat] <= empirical99pct))
                   
                    sns.regplot(
                        x=feat,
                        y=y_col,
                        data=this_df.loc[fil_outliers].sample(graph_sample, random_state=42),
                        scatter_kws={'alpha': .2},
                        lowess=True,
                    )
                    plt.title(f'{y_col} vs {feat}')
                    plt.show()
                except Exception as err:
                    print(err)
                    pass
