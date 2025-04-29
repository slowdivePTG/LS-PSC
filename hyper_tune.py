import pandas as pd
import numpy as np

if __name__ == "__main__":

    import argparse
    import os
    import json

    from dataset import DataSet
    from cv import Kfold_CV

    parser = argparse.ArgumentParser(
        prog="hyper_tune.py",
        description="Running cross validation for XGBoost models containing different features for hyperparameter tuning",
    )
    parser.add_argument("--model", help="model type (fiducial, apphot, white)", default="fiducial")
    parser.add_argument("--filters", help="filters to include", default="grz")
    parser.add_argument("--output", help="name of the output json file")
    parser.add_argument("--eta", help="learning rate", type=float, default=0.02)
    parser.add_argument("--n_estimators", help="number of trees", type=int, default=1000)
    parser.add_argument("--weight", help="weight of the hybrid model", type=float, default=0.3)
    parser.add_argument("--n_jobs", help="number of jobs", type=int, default=5)

    args = parser.parse_args()

    # hyperparameters
    hyper_fiducial = dict(
        n_estimators=args.n_estimators,
        eta=args.eta,
        max_depth=10,
        min_child_weight=6,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    hyper_range = dict(
        max_depth=[6, 8, 10, 12, 14, 16, 18],
        min_child_weight=[4, 6, 8, 10, 12],
        gamma=[0.1, 0.3, 0.5, 1.0, 2.0],
        subsample=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        colsample_bytree=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    # Round 0 - default parameters
    param_grid = dict()

    # Round 1 - tune max_depth and min_child_weight
    # param_grid = dict(
    #     max_depth=hyper_range["max_depth"],
    #     min_child_weight=hyper_range["min_child_weight"],
    # )

    # Round 2 - tune gamma
    # param_grid = dict(gamma=hyper_range["gamma"])

    # Round 3 - tune subsample, colsample_bytree
    # param_grid = dict(
    #     subsample=hyper_range["subsample"],
    #     colsample_bytree=hyper_range["colsample_bytree"],
    # )

    # training set
    ts_bright = pd.read_parquet("./Data/COSMOS/ls_dr10_cosmos_bright_relabel.parquet")
    ts = DataSet(
        ds=ts_bright,
        y_true=(ts_bright.HST_type == "STAR").values,
    )

    ##### features #####

    ##### cross validation #####
    # parameter grid

    job_dict = {"N_splits": 5, "N_jobs": args.n_jobs}

    #############################
    if args.model == "fiducial":
        print("Fiducial XGBoost model")
        X_features = ts.X

    #############################
    if args.model == "apphot":
        print("Apeture photometry")
        # normalized dchisq + AP
        X_features = ts.X_AP[args.filters]
        # hyper_fiducial["max_depth"] = 18
        # hyper_fiducial["min_child_weight"] = 10
        # hyper_fiducial["gamma"] = 0.1

    #############################
    if args.model == "white":
        print("Aperture photometry (white)")
        # normalized dchisq + AP (white)
        X_features = ts.X_white
        # hyper_fiducial["max_depth"] = 12
        # hyper_fiducial["min_child_weight"] = 6
        # hyper_fiducial["gamma"] = 1.0

    if args.model == "white_b+r":
        print("Aperture photometry (blue + red, trained separately)")
        X_features = ts.weighted_X_white_br
        hyper_fiducial["weight"] = args.weight  # CV results
        hyper_fiducial["max_depth"] = 12
        hyper_fiducial["min_child_weight"] = 6
        hyper_fiducial["gamma"] = 0.5
        # hyper_fiducial["subsample"] = 0.6
        # hyper_fiducial["colsample_bytree"] = 0.8

    # evaluation sets
    if args.model == "white_b+r":
        classifier = "WeightedXGBoost"
    else:
        classifier = "XGBoost"
    params_res, scores = Kfold_CV(
        X_feature=X_features,
        y_true=ts.y_true,
        classifier=classifier,
        param=hyper_fiducial,
        param_grid=param_grid,
        **job_dict,
    )
    res = {
        "model": args.model,
        "res": params_res,
    }
    res["param"] = hyper_fiducial
    res["param_grid"] = param_grid
    if args.model == "apphot":
        res["filters"] = args.filters

    # Save the optimal parameters and metrics
    output = f"cv_result_FoM/{args.output}.json"
    if len(os.listdir(output.split("/")[0])) == 0:
        os.system(f"touch {output}")

    with open(output, "w") as f:
        json.dump(res, f)

    # Save the scores
    np.save(f"cv_result_FoM/{args.output}.npy", scores)
