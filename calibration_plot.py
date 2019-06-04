def calibration_plot(df: pd.DataFrame, prob_col: str='pred_prob', outcome_col:str='clf', n_quant:int=20, addl_text="") -> None: 
    '''

    Generate a calibration plot. 
    This displays how well calibrated predictions are 
    (how our predicted probabilities compare to actual probabilities).
    (This is similar to a q-q plot.)

    df: pd.DataFrame, containing a column for predicted probability and a label.
    prob_col: str, label of column containing predicted probabilities in range [0.0, 1.0]
    outcome_col: str, label of column containing labels, 0 or 1
    n_quant: int, default=20
        Number of cuts we want to group the data into. In the dataframe, group 0 will
        represent the smallest predictions and group n_quant - 1 will be the bucket
        for the largest predictions
    addl_text: str, additional text for the graph title

    '''
    labels = [n / float(n_quant) for n in range(n_quant)]
    df['quant'] = pd.qcut(df[prob_col], n_quant, labels=labels)
    lift_df = df.groupby('quant').agg({prob_col:'mean', outcome_col:'mean'})
    plt.figure(figsize=(12,8))
    plt.plot(lift_df[prob_col].values, 'o-', linewidth=3)
    plt.plot(lift_df[outcome_col].values, 'o-', linewidth=3)
    plt.xlabel("Quantile")
    plt.ylabel("Probability")
    lift = round(
        (lift_df.iloc[n_quant - 1, 1] / lift_df.iloc[0, 1]), 4
    )
    print(f'Lift {lift}.')
    plt.title(f"Calibration of Predicted vs. Actual Probabilities {addl_text}")

    ticks = range(0, n_quant, 2)
    plt.xticks(ticks, labels[::2])

    plt.legend(['Predicted','Actual'])
    plt.show()
