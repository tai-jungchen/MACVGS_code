"""
Author: Alex (Tai-Jung) Chen

The python module that carries out multiple parameter tuning methods such as MADM, MADMCV, Grid search, Grid search CV,
and no tuning.
"""
import itertools
import logging
import math
import warnings
import numpy as np
from imblearn.metrics import specificity_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, \
    precision_recall_fscore_support, cohen_kappa_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from pyDecision.algorithm import bw_method
from sklearn.metrics import confusion_matrix
from typing import List
from sklearn.base import clone


class BaseClass:
    @staticmethod
    def npv_score(truth: np.ndarray, pred: np.ndarray) -> float:
        """
        Helper function for calculating the negative predictive value.

        :param truth: ground truth
        :param pred: prediction

        :return: negative predictive value
        """
        tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
        npv = tn / (tn + fn)
        if math.isnan(npv):
            warnings.warn("Negative Predicted Value is ill-defined and being set to 0.0 due to no predicted "
                          "negative samples.", category=UserWarning)
            return 0.0
        return npv


class MADMCVTuner(BaseClass):
    def __init__(self, model: object, metric_lst: List[str], mic: np.ndarray, lic: np.ndarray, hyps: dict,
                 thres: np.ndarray, cv: int = 10, stratify: bool = True, shuffle: bool = True):
        """
        Constructor

        :param model: model to be tuned.
        :param metric_lst: metrics that will be used to calculate the objective value for determine the optimality.
        :param mic: most important criteria in MADM algorithm.
        :param lic: least important criteria in MADM algorithm.
        :param hyps: hyperparameters to be tuned.
        :param thres: a list of decision thresholds to be tuned.

        :param cv: number of folds for cross validation
        :param stratify: stratified sampling or not
        :param shuffle: shuffle the data while performing CV or not
        """
        self.model = model
        self.metric_lst = metric_lst
        self.mic = mic
        self.lic = lic
        self.hyps = hyps
        self.thres = thres
        self.cv = cv
        self.strat = stratify
        self.shuffle = shuffle

        self.opt_param = None
        self.opt_thre = None
        self.opt_value = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Tune the given model with the input hyperparameters and thresholds.
        :param X: features
        :param y: label
        """
        # logging.info(f"MADM CV Tuning...")

        # use bwm to decide the weight
        weights = bw_method(self.mic, self.lic, eps_penalty=1, verbose=True)
        # get the optimal variable and value
        self.optimizer(X, y, weights)
        # fit the best model
        self.model.set_params(**self.opt_param).fit(X, y)

    def optimizer(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """
        Carry out the tuning on all hyperparameter, threshold pairs and store the optimal variables and value.
        Fill the objective value to the z table for later comparison use purpose.

        :param X: training features
        :param y: training labels
        :param w: weight
        """
        # Get all combinations of hyperparameter values
        keys, values = zip(*self.hyps.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # config for cv data splitting
        if self.strat:
            kf = StratifiedKFold(n_splits = self.cv, shuffle = self.shuffle, random_state=42)
        else:
            kf = KFold(n_splits=self.cv, shuffle=self.shuffle, random_state=42)

        # Iterate through each hyperparameter combination
        z_means = np.full((len(hyperparam_combinations), len(self.thres)), -1, dtype=float)
        param = 0
        for params in hyperparam_combinations:
            model = clone(self.model)
            model.set_params(**params)
            zs = np.full((self.cv, len(self.thres)), -1, dtype=float)

            # cv
            thre = 0
            for train_index, test_index in kf.split(X, y):
                X_train, X_test, y_train, y_test = X[train_index, :], X[test_index, :], y[train_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = self.predict(model, X_test)

                # per cv
                zs[thre, :] = self.score(y_pred, y_test, w)
                thre += 1

            # per param
            z_means[param, :] = np.mean(zs, axis=0)
            param += 1

        # Find optimal param and value
        index_flat = np.argmax(z_means)
        param_idx, thre_idx = np.unravel_index(index_flat, z_means.shape)
        self.opt_param = hyperparam_combinations[param_idx]
        self.opt_thre = self.thres[thre_idx]
        self.opt_value = np.max(z_means)

    def score(self, pred: np.ndarray, truth: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Returns the objective values for each hyperparameter, decision threshold combination. The objective value is
        the weighted sum of the classification metrics, where the weight is given by the BW method.

        :param pred: the predicted values
        :param truth: the truth
        :param weight: the weight calculated by bw method

        :return: objective values for each hyperparameter, decision threshold combination.
        """
        # accs = [accuracy_score(truth, row) for row in pred]
        # kappas = [cohen_kappa_score(truth, row) for row in pred]
        # baccs = [balanced_accuracy_score(truth, row) for row in pred]
        # f1_scores = [f1_score(truth, row) for row in pred]
        precisions = [precision_score(truth, row) for row in pred]
        recalls = [recall_score(truth, row) for row in pred]
        specs = [specificity_score(truth, row) for row in pred]
        npvs = [self.npv_score(truth, row) for row in pred]

        # aggregate metrics for weighted sum
        metrics = []
        for met in self.metric_lst:
            if met == "precision":
                metrics.append(precisions)
            elif met == "recall":
                metrics.append(recalls)
            elif met == "specificity":
                metrics.append(specs)
            elif met == "npv":
                metrics.append(npvs)
            else:
                raise Exception("Invalid classification metric!")

        metrics = np.array(metrics)
        if np.isnan(metrics).any():
            raise Exception("Classification metrics include NaN.")
        return np.dot(metrics.T, weight)

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """
        Return the predction based on the given decision threshold

        :param model: model generating the prediction.
        :param X: the features used to predict.

        :return: the predicted outcome based on the given decision threshold
        """
        if (isinstance(model, LogisticRegression) or isinstance(model, SVC) or isinstance(model, DecisionTreeClassifier)
            or isinstance(model, KNeighborsClassifier)) or isinstance(model, RandomForestClassifier) or isinstance(
        model, xgb.XGBClassifier):
            y_prob = np.tile(model.predict_proba(X)[:, 1], (len(self.thres), 1))
            y_pred = (y_prob >= self.thres[:, np.newaxis]).astype(int)
        else:
            raise Exception("Invalid model type! Please use one of the options in [LR, RF, ...] as input model type.")
        return y_pred


class MADMTuner(MADMCVTuner):
    def __init__(self, model: object, metric_lst: List[str], mic: np.ndarray, lic: np.ndarray, hyps: dict,
                 thres: np.ndarray, stratify: bool = True, shuffle: bool = True):
        """
        Constructor

        :param model: model to be tuned.
        :param metric_lst: metrics that will be used to calculate the objective value for determine the optimality.
        :param mic: most important criteria in MADM algorithm.
        :param lic: least important criteria in MADM algorithm.
        :param hyps: hyperparameters to be tuned.
        :param thres: a list of decision thresholds to be tuned.

        :param stratify: stratified sampling or not
        :param shuffle: shuffle the data while performing CV or not
        """
        super().__init__(model, metric_lst, mic, lic, hyps, thres, cv = 1, stratify = stratify, shuffle = shuffle)

    def optimizer(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """
        Carry out the tuning on all hyperparameter, threshold pairs and store the optimal variables and value.
        Fill the objective value to the z table for later comparison use purpose. Note that hte difference between
        this method and the identical named method in the parent class will be that the one in the parent class can
        deal with the Cross Validation process, but this one is designed for tuning without Cross Validation.

        :param X: training features
        :param y: training labels
        :param w: weight
        """
        # Get all combinations of hyperparameter values
        keys, values = zip(*self.hyps.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Train test split
        if self.strat:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        # Iterate through each hyperparameter combination
        z_means = np.full((len(hyperparam_combinations), len(self.thres)), -1, dtype=float)
        param = 0
        for params in hyperparam_combinations:
            model = clone(self.model)
            model.set_params(**params)

            model.fit(X_train, y_train)
            y_pred = self.predict(model, X_test)

            # per param
            z_means[param, :] = self.score(y_pred, y_test, w)
            param += 1

        # Find optimal param and value
        index_flat = np.argmax(z_means)
        param_idx, thre_idx = np.unravel_index(index_flat, z_means.shape)
        self.opt_param = hyperparam_combinations[param_idx]
        self.opt_thre = self.thres[thre_idx]
        self.opt_value = np.max(z_means)


class GridSearchCVTuner(BaseClass):
    def __init__(self, model: object, hyps: dict, thres: np.ndarray, metric: str, cv: int = 10, stratify: bool = True,
                 shuffle: bool = True):
        """
        Constructor

        :param model: model to be tuned.
        :param hyps: hyperparameters to be tuned.
        :param thres: a list of decision thresholds to be tuned.
        :param metric: classification metric used to determine optimality.

        :param cv: number of folds in cross validation
        :param stratify: stratified sampling or not
        :param shuffle: shuffle the data while performing CV or not
        """
        self.model = model
        self.hyps = hyps
        self.thres = thres
        self.metric = metric
        self.cv = cv
        self.strat = stratify
        self.shuffle = shuffle

        self.opt_param = None
        self.opt_thre = None
        self.opt_value = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Tune the given model with the input hyperparameters and thresholds.
        :param X: features
        :param y: label
        """

        # get the optimal variable and value
        self.optimizer(X, y)
        # fit the best model
        self.model.set_params(**self.opt_param).fit(X, y)

    def optimizer(self, X: np.ndarray, y: np.ndarray):
        """
        Carry out the tuning on all hyperparameter, threshold pairs and store the optimal variables and value.
        Fill the objective value to the z table for later comparison use purpose.

        :param X: training features
        :param y: training labels
        """
        # Get all combinations of hyperparameter values
        keys, values = zip(*self.hyps.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # config for cv data splitting
        if self.strat:
            kf = StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle, random_state=42)
        else:
            kf = KFold(n_splits=self.cv, shuffle=self.shuffle, random_state=42)

        # Iterate through each hyperparameter combination
        z_means = np.full((len(hyperparam_combinations), len(self.thres)), -1, dtype=float)
        param = 0
        for params in hyperparam_combinations:
            model = clone(self.model)
            model.set_params(**params)
            zs = np.full((self.cv, len(self.thres)), -1, dtype=float)

            # cv
            thre = 0
            for train_index, test_index in kf.split(X, y):
                X_train, X_test, y_train, y_test = X[train_index, :], X[test_index, :], y[train_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = self.predict(model, X_test)

                # per cv
                zs[thre, :] = self.score(y_pred, y_test)
                thre += 1

            # per param
            z_means[param, :] = np.mean(zs, axis=0)
            param += 1

        # Find optimal param and value
        index_flat = np.argmax(z_means)
        param_idx, thre_idx = np.unravel_index(index_flat, z_means.shape)
        self.opt_param = hyperparam_combinations[param_idx]
        self.opt_thre = self.thres[thre_idx]
        self.opt_value = np.max(z_means)

    def score(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        Returns the objective values for each hyperparameter, decision threshold combination. The criteria metric is
        defined in the constructor (self.metric).

        :param pred: the predicted values
        :param truth: the truth

        :return: objective values for each hyperparameter, decision threshold combination.
        """
        if self.metric == "accuracy":
            metrics = [accuracy_score(truth, row) for row in pred]
        elif self.metric == "kappa":
            metrics = [cohen_kappa_score(truth, row) for row in pred]
        elif self.metric == "balanced accuracy":
            metrics = [balanced_accuracy_score(truth, row) for row in pred]
        elif self.metric == "f1":
            metrics = [f1_score(truth, row) for row in pred]
        elif self.metric == "precision":
            metrics = [precision_score(truth, row) for row in pred]
        elif self.metric == "recall":
            metrics = [recall_score(truth, row) for row in pred]
        elif self.metric == "specificity":
            metrics = [specificity_score(truth, row) for row in pred]
        elif self.metric == "npv":
            metrics = [self.npv_score(truth, row) for row in pred]
        else:
            raise Exception("Unsupportive classification metric type.")

        metrics = np.array(metrics)
        if np.isnan(metrics).any():
            raise Exception("Classification metrics include NaN.")
        return metrics

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """
        Return the predction based on the given decision threshold

        :param model: model generating the prediction.
        :param X: the features used to predict.

        :return: the predicted outcome based on the given decision threshold
        """
        if (isinstance(model, LogisticRegression) or isinstance(model, SVC) or isinstance(model, DecisionTreeClassifier)
            or isinstance(model, KNeighborsClassifier)) or isinstance(model, RandomForestClassifier) or isinstance(
        model, xgb.XGBClassifier):
            y_prob = np.tile(model.predict_proba(X)[:, 1], (len(self.thres), 1))
            y_pred = (y_prob >= self.thres[:, np.newaxis]).astype(int)
        else:
            raise Exception("Invalid model type! Please use one of the options in [LR, RF, ...] as input model type.")
        return y_pred


class GridSearchTuner(GridSearchCVTuner):
    def __init__(self, model: object, hyps: dict, thres: np.ndarray, metric: str, stratify: bool = True,
                 shuffle: bool = True):
        """
        Constructor

        :param model: model to be tuned.
        :param hyps: hyperparameters to be tuned.
        :param thres: a list of decision thresholds to be tuned.
        :param metric: classification metric used to determine optimality.

        :param stratify: stratified sampling or not
        :param shuffle: shuffle the data while performing CV or not
        """
        super().__init__(model, hyps, thres, metric, cv=1, stratify=stratify, shuffle=shuffle)

    def optimizer(self, X: np.ndarray, y: np.ndarray):
        """
        Carry out the tuning on all hyperparameter, threshold pairs and store the optimal variables and value.
        Fill the objective value to the z table for later comparison use purpose. Note that the difference between
        this method and the identical named method in the parent class will be that the one in the parent class can
        deal with the Cross Validation process, but this one is designed for tuning without Cross Validation.

        :param X: training features
        :param y: training labels
        """
        # Get all combinations of hyperparameter values
        keys, values = zip(*self.hyps.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Train test split
        if self.strat:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        # Iterate through each hyperparameter combination
        z_means = np.full((len(hyperparam_combinations), len(self.thres)), -1, dtype=float)
        param = 0
        for params in hyperparam_combinations:
            model = clone(self.model)
            model.set_params(**params)

            model.fit(X_train, y_train)
            y_pred = self.predict(model, X_test)

            # per param
            z_means[param, :] = self.score(y_pred, y_test)
            param += 1

        # Find optimal param and value
        index_flat = np.argmax(z_means)
        param_idx, thre_idx = np.unravel_index(index_flat, z_means.shape)
        self.opt_param = hyperparam_combinations[param_idx]
        self.opt_thre = self.thres[thre_idx]
        self.opt_value = np.max(z_means)