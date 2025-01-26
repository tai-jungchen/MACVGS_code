"""
Author: Alex (Tai-Jung) Chen

Utilize MADM technique to perform parameter and threshold tuning. Dataset options include Home Equity dataset (HMEQ),
and COVID dataset (COVID). Model type options include Logistic Regression (LR), Random Forest (RF), Decision Tree (
DT), K Nearest Neighbor (KNN), Support Vector Machine (SVM), eXtreme Gradient Boosting (XGB).
"""
import numpy as np
import pandas as pd
import time
from imblearn.metrics import specificity_score
from sklearn.metrics import (accuracy_score, cohen_kappa_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from param_tuner import MADMCVTuner, MADMTuner, GridSearchCVTuner, GridSearchTuner, BaseClass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def main(dataset: str, model: object, hyps: dict, thres: np.ndarray, tuning: str, metric: str, metric_lst: list,
         mic: np.ndarray, lic: np.ndarray, n_rep: int = 2) -> dict:
    """
    Execute MADM tuning on given dataset, model type, mic, and lic.

    :param dataset: dataset to tune.
    :param model: model to tune.
    :param hyps: hyperparameter settings.
    :param thres: list of decision thresholds to be tuned.
    :param tuning: tuning type.
    :param metric: the classification metric used to determine optimality for grid search and grid search CV
    :param metric_lst: list of metrics used to calculate the madm objective value
    :param mic: Monotonicity Intensity Coefficient.
    :param lic: Linear Independence Condition.
    :param n_rep:  number of replication. Default set to 2.

    :return: summary result of all the replication including mean and s.e.
    """
    # dataset
    if dataset == "COVID":
        X = pd.read_pickle('datasets/pickles/covid_X.pkl').to_numpy()
        y = pd.read_pickle('datasets/pickles/covid_Y.pkl').to_numpy()
    elif dataset == "HMEQ":
        X = pd.read_pickle('datasets/pickles/hmeq_X.pickle')
        y = pd.read_pickle('datasets/pickles/hmeq_Y.pickle')
    elif dataset == "NIJ":
        data =pd.read_pickle("datasets/Raw_data/Global_MI_multi_cat_2.pkl")
        X = data.iloc[:, 1:-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
    else:
        raise Exception("Invalid dataset! Please use either COVID or HMEQ as input dataset.")

    # n replication
    record_metrics = ['acc', 'kappa', 'bacc', 'precision', 'recall', 'specificity', 'f1', 'npv', 'auc', 'apr', 'time']
    over_all_res = {key: [] for key in record_metrics}
    for i in range(n_rep):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=i)

        # Tuning type
        if tuning == "MACVGS":
            tuner = MADMCVTuner(model, metric_lst, mic, lic, hyps, thres, cv=10)
        elif tuning == "MADM":
            tuner = MADMTuner(model, metric_lst, mic, lic, hyps, thres)
        elif tuning == "GS":
            tuner = GridSearchTuner(model, hyps, thres, metric)
        elif tuning == "GSCV":
            tuner = GridSearchCVTuner(model, hyps, thres, metric, cv=10)
        else:
            raise Exception("Invalid tuning type!")

        # tune the model
        start_time = time.time()
        tuner.fit(X_train, y_train)
        end_time = time.time()
        y_pred, y_prob = predict(tuner.model, X_test, tuner.opt_thre)

        # classification performance
        over_all_res['acc'].append(round(accuracy_score(y_test, y_pred), 4))
        over_all_res['kappa'].append(round(cohen_kappa_score(y_test, y_pred), 4))
        over_all_res['bacc'].append(round(balanced_accuracy_score(y_test, y_pred), 4))
        over_all_res['precision'].append(round(precision_score(y_test, y_pred), 4))
        over_all_res['recall'].append(round(recall_score(y_test, y_pred), 4))
        over_all_res['specificity'].append(round(specificity_score(y_test, y_pred), 4))
        over_all_res['f1'].append(round(f1_score(y_test, y_pred), 4))
        over_all_res['npv'].append(round(BaseClass.npv_score(y_test, y_pred), 4))
        over_all_res['auc'].append(round(roc_auc_score(y_test, y_prob), 4))
        over_all_res['apr'].append(round(average_precision_score(y_test, y_prob), 4))
        over_all_res['time'].append(round(end_time - start_time, 4))

    # summarize results in all replication runs
    sum_res = {}
    sum_res['tune'] = [tuning]
    sum_res['model'] = [model]
    for key, val in over_all_res.items():
        mean_key = f"{key}_mean"
        se_key = f"{key}_se"
        val = np.array(val)
        sum_res[mean_key] = [round(float(np.mean(val)), 4)]
        sum_res[se_key] = [round(np.std(val, ddof=1) / np.sqrt(n_rep), 4)]
    return sum_res


def predict(model: object, X: np.ndarray, thre: float) -> tuple:
    """
    Return the prediction based on the given decision threshold

    :param model: model generating the prediction.
    :param X: the features used to predict.
    :param thre: decision threshold

    :return: the predicted outcome based on the given decision threshold and the predicted probability
    """
    if (isinstance(model, LogisticRegression) or isinstance(model, SVC) or isinstance(model, DecisionTreeClassifier)
            or isinstance(model, KNeighborsClassifier)) or isinstance(model, RandomForestClassifier) or isinstance(model, xgb.XGBClassifier):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        raise Exception("Invalid model type!")
    return (y_prob >= thre).astype(int), y_prob


if __name__ == '__main__':
    '''HMEQ case 1'''
    # data = "HMEQ"
    # scenario = "case1"
    # mic = np.array([3, 9, 1, 4])
    # lic = np.array([5, 1, 9, 5])
    # metric = "specificity"

    '''COVID case 1 - Infection rate'''
    # data = "COVID"
    # scenario = "case1"
    # mic = np.array([3, 1, 8, 5])
    # lic = np.array([6, 8, 1, 4])
    # metric = "recall"

    '''COVID case 2 - Budget'''
    # data = "COVID"
    # scenario = "case2"
    # mic = np.array([3, 8, 1, 4])
    # lic = np.array([6, 1, 8, 5])
    # metric = "specificity"

    '''NIJ case 1'''
    data = "NIJ"
    scenario = "case1"
    mic = np.array([8, 3, 6, 1])
    lic = np.array([1, 5, 4, 8])
    metric = "npv"

    metric_lst = ["precision", "recall", "specificity", "npv"]
    tunes = ["MACVGS", "GSCV"]  # tunes = ["MACVGS", "GSCV", "MADM", "GS"]
    model_types = ["LR", "DT", "KNN", "RF", "XGB"]
    thresholds = np.arange(0, 1.05, 0.05)

    # iterate through all configurations
    df = pd.DataFrame()
    for tune in tunes:
        for model_type in model_types:
            if model_type == "LR":
                input_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100)
                hyper_param = {"C": [1000, 100, 10, 1, 0.1, 0.01, 0.001], "class_weight": ['balanced', None]}
            elif model_type == "SVM":
                input_model = SVC(probability=True)
                hyper_param = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], "class_weight": ['balanced', None]}
            elif model_type == "DT":
                input_model = DecisionTreeClassifier()
                hyper_param = {'max_depth': [10, 50, 100], "class_weight": ['balanced', None],
                               "criterion": ["gini", "entropy", "log_loss"]}
            elif model_type == "KNN":
                input_model = KNeighborsClassifier()
                hyper_param = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
            elif model_type == "RF":
                input_model = RandomForestClassifier()
                hyper_param = {'n_estimators': [100, 200, 300], 'class_weight': [None, 'balanced']}
            elif model_type == "XGB":
                input_model = xgb.XGBClassifier()
                hyper_param = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.3]}
            else:
                raise Exception("Unsupported model type.")

            final_res = main(data, input_model, hyper_param, thresholds, tune, metric, metric_lst, mic, lic, n_rep=2)
            sub_df = pd.DataFrame(final_res)
            df = pd.concat([df, sub_df], axis=0)

    filename = f"{data}_{scenario}_{metric}.csv"
    df.to_csv(filename, index=False)

