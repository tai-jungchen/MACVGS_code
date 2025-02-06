"""
Author: Alex (Tai-Jung) Chen

Pull out the weights calculated by bwm algorithm.
"""

import numpy as np
from pyDecision.algorithm import bw_method

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

weights = bw_method(mic, lic, eps_penalty=1, verbose=True)
metric_lst = ["precision", "recall", "specificity", "npv"]

print(f"{data}: {weights} on metric list {metric_lst}")