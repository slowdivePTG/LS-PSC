import numpy as np
import pandas as pd

import argparse

from dataset import DataSet
from cv import Kfold_CV, GridSearchTest
from load_desi import load_desi

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="cross_validation_COSMOS.py",
        description="Running cross validation for RF/XGBoost models containing different features",
    )
    parser.add_argument(
        "--model", help="model type (fiducial, apphot, white)", default="fiducial"
    )
    parser.add_argument(
        "--scoring", help="scoring metric", default="FoM", choices=["roc_auc", "FoM"]
    )
    parser.add_argument("--filters", help="filters to include", default="grz")
    parser.add_argument("--output", help="name of the output json file")
    parser.add_argument(
        "--method",
        help="validate on the training set (cross validation) or the test set",
        default="cv",
    )

    args = parser.parse_args()

    # hyperparameters
    hyper_fiducial = dict(
        n_estimators=500,
        eta=0.05,
        max_depth=10,
        min_child_weight=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0,  # 0.1,
        reg_lambda=0,  # 0.1,
    )
    hyper_range = dict(
        max_depth=[6, 8, 10, 12],
        min_child_weight=[4, 6, 8, 10, 12],
        gamma=[0.1, 0.3, 0.5, 1],
        reg_alpha=[0, 0.1, 0.3],
        reg_lambda=[0.2, 0.3, 0.4, 0.5],
        subsample=[0.6, 0.7, 0.8, 0.9],
        colsample_bytree=[0.6, 0.7, 0.8, 0.9],
        # subsample=[0.8, 0.85, 0.9, 0.95],
        # colsample_bytree=[0.7, 0.75, 0.8],
    )

    # Round 1 - tune max_depth
    # param_grid = dict(
    #     max_depth=hyper_range["max_depth"],
    # )

    # Round 2 - tune min_child_weight, subsample, colsample_bytree
    # param_grid = dict(
    #     min_child_weight=hyper_range["min_child_weight"],
    # )

    # Round 3 - tune subsample, colsample_bytree
    # param_grid = dict(
    #     subsample=hyper_range["subsample"],
    #     colsample_bytree=hyper_range["colsample_bytree"],
    # )

    # Round 4 - tune gamma
    param_grid = dict(
        gamma=[0.1], #hyper_range["gamma"],
    )

    # training set
    ts_bright = pd.read_parquet("./Data/ls_dr10_cosmos_bright_relabel.parquet")
    print(f"Number of training objects: {len(ts_bright)}")
    ts = DataSet(
        ds=ts_bright,
        y_true=(ts_bright.HST_type == "STAR").values,
    )

    ##### features #####

    ##### cross validation #####
    # parameter grid

    job_dict = {"N_splits": 5, "N_jobs": 5}

    #############################
    if args.model == "fiducial":
        print("Fiducial XGBoost model")
        X_features = ts.X

    #############################
    if args.model == "apphot":
        print("Apeture photometry")
        # normalized dchisq + AP
        X_features = ts.X_AP[args.filters]
        hyper_fiducial["max_depth"] = 8
        hyper_fiducial["min_child_weight"] = 8
        hyper_fiducial["colsample_bytree"] = 0.7

    #############################
    if args.model == "white":
        print("Aperture photometry (white)")
        # normalized dchisq + AP (white)
        X_features = ts.X_white
        hyper_fiducial["min_child_weight"] = 8
        hyper_fiducial["gamma"] = 0.5

    if args.model == "white_br":
        print("Aperture photometry (blue + red)")
        # normalized dchisq + AP (blue + red)
        X_features = ts.X_white_br

    if args.model == "white_b+r":
        print("Aperture photometry (blue + red, trained separately)")
        X_features = ts.weighted_X_white_br
        hyper_fiducial["weight"] = 0.4  # CV results
        hyper_fiducial["min_child_weight"] = 10
        hyper_fiducial["colsample_bytree"] = 0.9
        hyper_fiducial["subsample"] = 0.9

    if args.method == "test":
        # test set
        desi_clean = load_desi(verbose=False)
        print(f"Number of test objects: {len(desi_clean)}")
        test = DataSet(
            ds=desi_clean,
            y_true=(desi_clean.spectype != "GALAXY").values,
        )
        #############################
        if args.model == "fiducial":
            print("Fiducial XGBoost model")
            X_features_test = test.X

        #############################
        if args.model == "apphot":
            print("Apeture photometry")
            # normalized dchisq + AP
            X_features_test = test.X_AP[args.filters]

        #############################
        if args.model == "white":
            print("Aperture photometry (white)")
            # normalized dchisq + AP (white)
            X_features_test = test.X_white

        if args.model == "white_br":
            print("Aperture photometry (blue + red)")
            # normalized dchisq + AP (blue + red)
            X_features_test = test.X_white_br

        if args.model == "white_b+r":
            print("Aperture photometry (blue + red, trained separately)")
            X_features_test = test.weighted_X_white_br
            hyper_fiducial["weight"] = 0.3  # CV results


    # evaluation sets
    if args.model == "white_b+r":
        classifier = "WeightedXGBoost"
    else:
        classifier = "XGBoost"

    if args.method == "cv":
        res = {
            "model": args.model,
            "res": Kfold_CV(
                X_feature=X_features,
                y_true=ts.y_true,
                classifier=classifier,
                param=hyper_fiducial,
                param_grid=param_grid,
                score=args.scoring,
                **job_dict,
            ),
        }
    elif args.method == "test":
        res = {
            "model": args.model,
            "res": GridSearchTest(
                X_feature_tr=X_features,
                y_true_tr=ts.y_true,
                X_feature_test=X_features_test,
                y_true_test=test.y_true,
                classifier=classifier,
                param=hyper_fiducial,
            ).Search(param_grid=param_grid, scoring=args.scoring, N_jobs=5),
        }
    res["param"] = hyper_fiducial
    res["param_grid"] = param_grid
    if args.model == "apphot":
        res["filters"] = args.filters

    import os

    output = f"cv_result_{args.scoring}/{args.output}.json"
    if len(os.listdir(output.split("/")[0])) == 0:
        os.system(f"touch {output}")

    import json

    with open(output, "w") as f:
        json.dump(res, f)
