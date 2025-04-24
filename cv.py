import numpy as np

# metric
from scipy.interpolate import interp1d

from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

# cv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer


class WeightedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that combines two XGBClassifiers with different weights
    """

    def __init__(self, clf1=None, clf2=None, weight=0.5, **params):
        self.eval_metric = params.pop("eval_metric", None)
        self.weight = weight
        if self.eval_metric is None:
            self.eval_metric = "logloss"
        self.early_stopping_rounds = params.pop("early_stopping_rounds", None)

        if clf1 is None:
            self.clf1 = XGBClassifier(**params)
        if clf2 is None:
            self.clf2 = XGBClassifier(**params)
        self.kwargs = self.clf1.get_params()

    def set_params(self, **params):
        self.clf1.set_params(**params)
        self.clf2.set_params(**params)
        if "weight" in params:
            self.weight = params["weight"]
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
        assert len(X_train_1[0, :]) == len(X_train_2[0, :]), "X_train_1 and X_train_2 have different shapes"

        if "eval_set" in kwargs:
            eval_set = kwargs.pop("eval_set")
            self.early_stopping_rounds = kwargs.pop("early_stopping_rounds", self.early_stopping_rounds)
            verbose = kwargs.pop("verbose", False)

            # Initialize variables for manual early stopping
            best_score = float("-inf")
            best_iteration = 0
            no_improvement_rounds = 0
            self.eval_results_ = {}

            # Initialize eval_results_ properly for both training and validation sets
            self.eval_results_ = {
                "validation_0": {self.eval_metric: []},  # Training set results
                "validation_1": {self.eval_metric: []}   # Validation set results
            }

            # Maximum number of boosting rounds
            max_rounds = self.kwargs.get("n_estimators", 1000)

            # Training in iterations
            for iteration in range(50, max_rounds + 1, 50):  # Step size of 50
                # Update number of estimators for both models
                self.clf1.n_estimators = iteration
                self.clf2.n_estimators = iteration

                # Fit both models without early stopping
                self.clf1.fit(X_train_1, y, eval_set=None)
                self.clf2.fit(X_train_2, y, eval_set=None)

                # Evaluate combined model performance on validation set
                train_score = self._calculate_metric(x=eval_set[0][0], y_true=eval_set[0][1])
                val_score = self._calculate_metric(x=eval_set[1][0], y_true=eval_set[1][1])

                # Store the scores
                self.eval_results_["validation_0"][self.eval_metric].append(train_score)
                self.eval_results_["validation_1"][self.eval_metric].append(val_score)

                if verbose:
                    print(f"[{iteration}] Combined validation score: {val_score:.4f}")

                # Track the best performance
                if val_score > best_score:
                    best_score = val_score
                    best_iteration = iteration
                    no_improvement_rounds = 0
                else:
                    no_improvement_rounds += 50

                # Early stopping check
                if self.early_stopping_rounds is not None and no_improvement_rounds >= self.early_stopping_rounds:
                    if verbose:
                        print(f"Early stopping triggered at iteration {iteration}")
                    break

            # Store best iteration
            self.best_iteration_ = best_iteration

            # Final fit with best iteration
            self.clf1.n_estimators = self.best_iteration_
            self.clf2.n_estimators = self.best_iteration_
            self.clf1.fit(X_train_1, y)
            self.clf2.fit(X_train_2, y)

        else:
            self.clf1.fit(X_train_1, y)
            self.clf2.fit(X_train_2, y)

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
            # "clf1": deepcopy(self.clf1),
            # "clf2": deepcopy(self.clf2),
            "weight": self.weight,
            **self.kwargs,
        }
        return params

    def _calculate_metric(self, x, y_true):
        from sklearn.metrics import (
            f1_score,
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
            log_loss,
        )

        # Based on probabilities
        if self.eval_metric in ["logloss", "roc_auc", "FoM"]:
            y_pred = self.predict_proba(x)[:, 1]
        # Based on labels
        elif self.eval_metric in ["f1", "accuracy", "precision", "recall"]:
            y_pred = self.predict(x)
        else:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric}")

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


def FoM_score(y_true, y_score, FPR=0.005):
    """Figure of merit"""

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score, drop_intermediate=False)

    return interp1d(x=fpr, y=tpr)(FPR).ravel()[0]


def Kfold_CV(
    X_feature,
    y_true,
    classifier="XGBoost",
    param={},
    param_grid={},
    N_splits=5,
    N_jobs=-2,
) -> tuple[dict, np.ndarray]:
    """
    K-fold cross validation composed of an inner and outer loop.
    The inner loop performs a grid search to find the best parameters, and the outer loop evaluates the model.

    Parameters
    ----------
    X_feature : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_true : np.ndarray
        True labels of shape (n_samples,).
    classifier : str
        Classifier type. Options are "RF", "XGBoost", or "WeightedXGBoost".
    param : dict
        Fixed parameters for the classifier.
    param_grid : dict
        Parameters for the grid search.
    N_splits : int
        Number of splits for K-fold cross validation.
    N_jobs : int
        Number of jobs to run in parallel. -1 means using all processors.

    Returns
    -------
    tuple[dict, np.ndarray]
        A tuple containing the results of the cross-validation and the scores for each sample.
    """
    kf = StratifiedKFold(n_splits=N_splits, shuffle=True, random_state=114514)
    best_params = []
    scores = np.zeros_like(y_true, dtype=float)
    accuracy, FoM, roc_auc = [], [], []
    for i, (train_idx, test_idx) in enumerate(kf.split(X=X_feature, y=y_true)):
        print(f"Fold {i}:")

        # Split the data into training and testing sets
        X_train, X_test = X_feature[train_idx], X_feature[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        # Train the model
        if classifier == "RF":
            clf_type = RandomForestClassifier()
        elif classifier == "XGBoost":
            clf_type = XGBClassifier()
        elif classifier == "WeightedXGBoost":
            clf_type = WeightedXGBClassifier()

        # Set the fixed parameters
        clf_type.set_params(**param)

        # The inner loop of the grid search
        clf = GridSearchCV(
            clf_type,
            param_grid=param_grid,
            verbose=3,
            cv=StratifiedKFold(n_splits=N_splits, shuffle=True, random_state=1919810),
            n_jobs=N_jobs,
            scoring=make_scorer(FoM_score, response_method="predict_proba"),
            error_score="raise",
        )

        clf.fit(X_train, y_train, verbose=False)

        best_params.append(clf.best_params_)
        print(clf.best_params_)

        test_pred = clf.best_estimator_.predict(X_test)
        test_pred_p = clf.best_estimator_.predict_proba(X_test)[:, 1]

        scores[test_idx] = test_pred_p

        accuracy.append(accuracy_score(y_true=y_test, y_pred=test_pred))
        FoM.append(FoM_score(y_true=y_test, y_score=test_pred_p))
        roc_auc.append(roc_auc_score(y_true=y_test, y_score=test_pred_p))

    print("Accuracy = {:.4f} +/- {:.4f}".format(np.mean(accuracy), np.std(accuracy)))
    print("FoM = {:.4f} +/- {:.4f}".format(np.mean(FoM), np.std(FoM)))
    print("ROC_AUC = {:.4f} +/- {:.4f}".format(np.mean(roc_auc), np.std(roc_auc)))

    res = {
        "classifier": classifier,
        "scoring": "FoM",
        "accuracy": accuracy,
        "FoM": FoM,
        "ROC_AUC": roc_auc,
        "best_params": best_params,
    }

    return res, scores
