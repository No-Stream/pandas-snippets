from bayes_opt import BayesianOptimization
from datetime import datetime
from tqdm import tqdm
# Keep optimizer logs since it's time-consuming.
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

N_INIT_ROUNDS = 30

N_OPTIMIZATION_ROUNDS = 200

dtrain = xgb.DMatrix(x_train, label=xs["outcome"])
dtest = xgb.DMatrix(x_test)


def print_optimizer_result(opt):
    res = sorted(opt.res, key=lambda dic: dic["target"], reverse=True)

    print("  N Bayesian optimization iterations: %i" % len(opt.res))

    print("\n  Baseline:")
    print(round(opt.res[0], 4))

    print("\n  Best:")
    print(round(opt.max, 4))

    print("\n\n")

    plt.figure(figsize=(15, 8))
    plt.plot([res["target"] for res in opt.res])
    plt.title("Optimizer AUC Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")


def xgb_evaluate_fine(
    max_depth,
    gamma,
    colsample_bytree,
    subsample,
    min_child_weight,
    reg_lambda,
    reg_alpha,
    eta,
    boost_rounds,
    scale_pos_weight,
    nfold=10,
):
    """

    Get a score for a single set of hyperparameters.
    Uses 10 stratified folds by default, can be droppped to 5 if your dataset is very large.

    """

    xgb_params = {
        "eval_metric": "auc",
        "max_depth": round(max_depth),
        "subsample": subsample,
        "eta": eta,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "scale_pos_weight": scale_pos_weight,
        "seed": 2016,
        "random_state": 2001,
        # Choice of threads here depends on your CPU.
        # Remember, CV parallelizes more linearly than boosting.
        "nthread": 1,
        "silent": True,
        "tree_method": "hist",  # faster, with less overfit than 'exact'
        "grow_policy": "depthwise",  # less overfit w/ hist vs. lossguide
    }
    # use correct-ish # of boost rounds
    cv_result = xgb.cv(
        xgb_params,
        dtrain,
        num_boost_round=int(boost_rounds),
        nfold=nfold,
        stratified=True,
        seed=2001,
        metrics="auc",
    )

    # Bayesian optimization only knows how to maximize, not minimize.
    gc.collect()
    return cv_result["test-auc-mean"].iloc[-1]


logger2 = JSONLogger(path="./bayesopt_logs.json")

param_ranges = {
    "max_depth": (1, 20),
    "gamma": (0.0, 0.05),
    "colsample_bytree": (0.60, 0.95),
    "subsample": (0.6, 0.98),
    "min_child_weight": (0.2, 2.0),
    "reg_lambda": (0.10, 100.0),
    "reg_alpha": (0.01, 50.0),
    # 'scale_pos_weight' can be set based on the class balance in your dataset, (# 0's class / # 1's class)
    #   but 1.0 will often deliver max AUC and accurate pred probs
    #'scale_pos_weight': (0.5, 2.0),
    "eta": (0.010, 0.30),
    "boost_rounds": (50, 3000),
}

defaults = {
    "max_depth": 6,
    "gamma": 0,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "min_child_weight": 1.0,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    #'scale_pos_weight': 1.0,
    "eta": 0.10,
    "boost_rounds": 200,
}

defaults_2 = {
    "max_depth": 8,
    "gamma": 0,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "min_child_weight": 1.0,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    #'scale_pos_weight': 1.0,
    "eta": 0.02,
    "boost_rounds": 1000,
}

defaults_2 = {
    "max_depth": 8,
    "gamma": 0,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "min_child_weight": 1.0,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    #'scale_pos_weight': 1.0,
    "eta": 0.01,
    "boost_rounds": 3000,
}


xgb_bo_fine = BayesianOptimization(
    f=xgb_evaluate_fine, pbounds=param_ranges, random_state=2001
)
xgb_bo_fine.subscribe(Events.OPTMIZATION_STEP, logger2)

print("\n\n  Try default parameters")
xgb_bo_fine.probe(params=defaults, lazy=False)
xgb_bo_fine.probe(params=defaults_2, lazy=False)
xgb_bo_fine.probe(params=defaults_3, lazy=False)
xgb_bo_fine.maximize(init_points=0, n_iter=0, acq="ei")  # default

print("\n\n  Initialize hyperparameter space.")
for iter in tqdm(range(N_INIT_ROUNDS)):
    gc.collect()
    xgb_bo_fine.maximize(init_points=1, n_iter=0, acq="ei")


print("\n\n  Search.")
for iter in tqdm(range(N_OPTIMIZATION_ROUNDS)):
    gc.collect()
    xgb_bo_fine.maximize(init_points=0, n_iter=1, acq="ei")


# Opt results
print_optimizer_result(xgb_bo_fine)
