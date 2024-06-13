import numpy as np

# metric
from scipy.interpolate import interp1d


def FoM_score(y_true, y_score, FPR=0.005):
    """Figure of merit"""

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score, drop_intermediate=False)

    return interp1d(x=fpr, y=tpr)(FPR).ravel()[0]


from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


class WeightedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that combines two XGBClassifiers with different weights
    """

    def __init__(self, clf1=None, clf2=None, weight=0.5, **params):
        self.weight = weight
        self.eval_metric = params.pop("eval_metric", None)
        if self.eval_metric is None:
            self.eval_metric = "logloss"
        self.early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        if clf1 is None:
            self.clf1 = XGBClassifier(**params)
        if clf2 is None:
            self.clf2 = XGBClassifier(**params)
        self.kwargs = self.clf1.get_params()

    def set_params(self, **params):
        self.clf1.set_params(**params)
        self.clf2.set_params(**params)
        self.weight = params.pop("weight", self.weight)
        self.kwargs = self.clf1.get_params()
        return self

    def fit(self, X, y, **kwargs):
        self.classes_ = np.array([0, 1])
        # First 2 columns indicate when each classifier is masked
        # The rest of the columns are the features
        X_train = X[:, 2:]
        # Last 7 * 2 columns correspond to aperture flux ratios
        X_train_1 = X_train[:, :-7]
        X_train_2 = np.concatenate([X_train[:, :-14], X_train[:, -7:]], axis=1)
        assert len(X_train_1[0, :]) == len(
            X_train_2[0, :]
        ), "X_train_1 and X_train_2 have different shapes"

        if "eval_set" in kwargs:
            eval_set = kwargs.pop("eval_set")
            self.early_stopping_rounds = kwargs.pop(
                "early_stopping_rounds", self.early_stopping_rounds
            )

            # Do early stopping manually
            best_score = 0
            best_iteration = 0
            no_improvement_rounds = 0

            self.eval_results_ = {}
            # Initialize the evaluation results
            for k in range(len(eval_set)):
                self.eval_results_["validation_{}".format(k)] = {}
                self.eval_results_["validation_{}".format(k)][self.eval_metric] = []

            for epoch in np.arange(self.kwargs["n_estimators"] // 25) * 25:
                # Train one epoch for both models
                self.clf1.n_estimators = epoch + 25
                self.clf2.n_estimators = epoch + 25
                self.clf1.fit(X_train_1, y, early_stopping_rounds=None)
                self.clf2.fit(X_train_2, y, early_stopping_rounds=None)

                # Predict the validation set
                if type(eval_set) is tuple:
                    eval_set = [eval_set]
                for k in range(len(eval_set)):
                    hybrid_preds = self.predict_proba(eval_set[k][0])[:, 1]
                    self.eval_results_["validation_{}".format(k)][
                        self.eval_metric
                    ].append(FoM_score(y_true=eval_set[k][1], y_score=hybrid_preds))

                # Check if the score has improved
                score = self.eval_results_["validation_1"][self.eval_metric][-1]
                if score > best_score:
                    best_score = score
                    best_iteration = epoch + 25
                    no_improvement_rounds = 0
                else:
                    no_improvement_rounds += 25
                if kwargs.get("verbose", 0) > 0:
                    print(
                        f"N_estimators {epoch + 25}: score = {score:.4f}, best score = {best_score:.4f}; no improvement rounds = {no_improvement_rounds}"
                    )

                # Early stopping
                if no_improvement_rounds >= self.early_stopping_rounds:
                    print(f"Early stopping at epoch {epoch + 25}")
                    break

            # Set the final number of estimators to the best iteration
            self.clf1.n_estimators = best_iteration
            self.clf1.n_estimators = best_iteration

        self.clf1.fit(X_train_1, y)
        self.clf2.fit(X_train_2, y)

        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        X_mask_clf1 = X[:, 0]
        X_mask_clf2 = X[:, 1]
        X_train = X[:, 2:]
        X_train_1 = X_train[:, :-7]
        X_train_2 = np.concatenate([X_train[:, :-14], X_train[:, -7:]], axis=1)
        weight_clf1 = (1 - X_mask_clf1) * self.weight
        weight_clf2 = (1 - X_mask_clf2) * (1 - self.weight)
        return (
            weight_clf1[:, np.newaxis] * self.clf1.predict_proba(X_train_1)
            + weight_clf2[:, np.newaxis] * self.clf2.predict_proba(X_train_2)
        ) / (weight_clf1 + weight_clf2)[:, np.newaxis]

    def predict(self, X):
        return np.array(self.predict_proba(X)[:, 1] > 0.5, dtype=bool)

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "weight": self.weight,
            **self.kwargs,
        }
        return params

    def _calculate_metric(self, y_true, y_pred):
        from sklearn.metrics import (
            f1_score,
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
            log_loss,
        )

        if self.eval_metric == "logloss":
            return log_loss(y_true, y_pred)
        if self.eval_metric == "f1":
            return f1_score(y_true, y_pred, average="macro")
        elif self.eval_metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif self.eval_metric == "precision":
            return precision_score(y_true, y_pred, average="macro")
        elif self.eval_metric == "recall":
            return recall_score(y_true, y_pred, average="macro")
        elif self.eval_metric == "roc_auc":
            return roc_auc_score(y_true, y_pred)
        elif self.eval_metric == "FoM":
            return FoM_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric}")


# cv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer

# multiprocessing
from multiprocessing import Pool


def Kfold_CV(
    X_feature,
    y_true,
    classifier="XGBoost",
    param={},
    param_grid={},
    score="FoM",
    N_splits=5,
    N_jobs=-2,
):
    """K-fold cross validation"""
    kf = StratifiedKFold(n_splits=N_splits, shuffle=True, random_state=114514)
    best_params = []
    accuracy, FoM, roc_auc = [], [], []
    if score == "FoM":
        scoring = make_scorer(FoM_score, response_method="predict_proba")
    elif score == "roc_auc":
        scoring = "roc_auc"
    for i, (train_idx, test_idx) in enumerate(kf.split(X=X_feature, y=y_true)):
        print(f"Fold {i}:")

        X_train, X_test = X_feature[train_idx], X_feature[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        if classifier == "RF":
            clf_type = RandomForestClassifier()
        elif classifier == "XGBoost":
            clf_type = XGBClassifier()
        elif classifier == "WeightedXGBoost":
            clf_type = WeightedXGBClassifier()
        clf_type.set_params(**param)
        clf = GridSearchCV(
            clf_type,
            param_grid=param_grid,
            verbose=3,
            cv=StratifiedKFold(n_splits=N_splits, shuffle=True, random_state=1919810),
            n_jobs=N_jobs,
            scoring=scoring,
            error_score="raise",
        )

        clf.fit(X_train, y_train, verbose=False)

        best_params.append(clf.best_params_)
        print(clf.best_params_)

        test_pred = clf.best_estimator_.predict(X_test)
        test_pred_p = clf.best_estimator_.predict_proba(X_test)[:, 1]

        accuracy.append(accuracy_score(y_true=y_test, y_pred=test_pred))
        FoM.append(FoM_score(y_true=y_test, y_score=test_pred_p))
        roc_auc.append(roc_auc_score(y_true=y_test, y_score=test_pred_p))

    print("Accuracy = {:.4f} +/- {:.4f}".format(np.mean(accuracy), np.std(accuracy)))
    print("FoM = {:.4f} +/- {:.4f}".format(np.mean(FoM), np.std(FoM)))
    print("ROC_AUC = {:.4f} +/- {:.4f}".format(np.mean(roc_auc), np.std(roc_auc)))

    return {
        "classifier": classifier,
        "scoring": score,
        "accuracy": accuracy,
        "FoM": FoM,
        "ROC_AUC": roc_auc,
        "best_params": best_params,
    }


class GridSearchTest:
    def __init__(
        self,
        X_feature_tr,
        y_true_tr,
        X_feature_test,
        y_true_test,
        classifier="XGBoost",
        param={},
    ):
        self.X_feature_tr = X_feature_tr
        self.y_true_tr = y_true_tr
        self.X_feature_test = X_feature_test
        self.y_true_test = y_true_test
        self.clf_type = classifier
        if classifier == "RF":
            self.clf = RandomForestClassifier()
        elif classifier == "XGBoost":
            self.clf = XGBClassifier()
        elif classifier == "WeightedXGBoost":
            self.clf = WeightedXGBClassifier()
        self.clf.set_params(**param)
        # print(self.clf.get_params())

    def evaluate_model(self, param):
        print(param)

        self.clf = self.clf.set_params(**param)
        X_tr, y_tr = self.X_feature_tr, self.y_true_tr
        X_test, y_test = self.X_feature_tr, self.y_true_tr
        self.clf.fit(X_tr, y_tr)

        test_pred_p = self.clf.predict_proba(X_test)[:, 1]
        test_pred = np.array(test_pred_p > 0.5, dtype=bool)

        acc = accuracy_score(y_true=y_test, y_pred=test_pred)
        fom = FoM_score(y_true=y_test, y_score=test_pred_p)
        auc = roc_auc_score(y_true=y_test, y_score=test_pred_p)

        print("Accuracy = {:.4f}".format(acc))
        print("FoM = {:.4f}".format(fom))
        print("ROC_AUC = {:.4f}".format(auc))

        return dict(param=param, accuracy=acc, FoM=fom, roc_auc=auc)

    def Search(self, param_grid, scoring, N_jobs=5):
        from sklearn.model_selection import ParameterGrid

        param_grid = list(ParameterGrid(param_grid))

        # Create a Pool for parallel execution
        with Pool(processes=N_jobs) as p:
            results = p.map(self.evaluate_model, param_grid)

        assert scoring in ["accuracy", "FoM", "roc_auc"], "Invalid scoring"
        best_results = max(results, key=lambda x: x[scoring])
        print("Best param:")
        print(best_results["param"])
        print("Accuracy = {:.4f}".format(best_results["accuracy"]))
        print("FoM = {:.4f}".format(best_results["FoM"]))
        print("ROC_AUC = {:.4f}".format(best_results["roc_auc"]))

        best_results["classifier"] = self.clf_type
        best_results["scoring"] = scoring
        best_results["param_grid"] = param_grid

        return best_results
