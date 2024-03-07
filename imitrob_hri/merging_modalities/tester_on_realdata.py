
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from imitrob_nlp.nlp_utils import make_conjunction
import data.datagen_utils2 as datagen_utils2 
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path



def tester_all(exp=['arity'], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=True, acto=False):

    data_real_folder = Path(__file__).parent.parent.joinpath('data_real')

    with open(data_real_folder.joinpath('dataset20240226_233209_res.log')) as f:
        y_file = f.read()
    y_file_lines = y_file.split("\n")
    
    ground_truth = []
    # ground_truth = trial (log line = some trial) x (experiment,n trial,template,selection,storage)
    for y_file_line in y_file_lines:
        y_line_elements = y_file_line.split(",")
        ground_truth.append(y_line_elements)

    all_ex = 0
    acc_bl = 0
    acc_m1 = 0
    acc_m2 = 0
    acc_m3 = 0
    for cn,c in enumerate(exp):
        for pn,d in enumerate(alignm):
            
            for tn,t in enumerate([1,2,3,4,5,6,7,8,9,10]): # trial 
                for sn,s in enumerate(['r', 'k', 'p']):

                    fld = data_real_folder.joinpath(c,d)
                    for file in fld.glob(f"{c}_{d}_{t}_{s}_mms_trial_*"): # all trials
                        n_of_trial = str(file)[:-4].split("_")[-1]

                        mms_trial = np.load(fld.joinpath(f"{c}_{d}_{t}_{s}_mms_trial_{n_of_trial}.npy"), allow_pickle=True)
                        str_trial = np.load(fld.joinpath(f"{c}_{d}_{t}_{s}_str_trial_{n_of_trial}.npy"), allow_pickle=True)


                        L = mms_trial[0].L
                        G = mms_trial[0].G
                        # search for a line, which corresponds to given experiment and trial
                        GT = []
                        for gt_line in ground_truth:
                            # print(f"{gt_line[0]} == {c}, {gt_line[0] == c} and {gt_line[1]} == {t}, {str(gt_line[1]) == str(t)}")
                            if str(gt_line[0]) == str(c) and str(gt_line[1]) == str(t):
                                GT = gt_line[2:]
                                break
                        # check gt_line found
                        assert len(GT)>0, f"line {c}, {t} not found in dataset20240226_233209_res.log"

                        BL = mms_trial[0].M
                        M1 = mms_trial[1].M
                        M2 = mms_trial[2].M
                        M3 = mms_trial[3].M

                        if compare(BL, GT, thresholds=False, action_only=acto): acc_bl += 1 
                        if compare(M1, GT, thresholds=thr, action_only=acto): acc_m1 += 1
                        if compare(M2, GT, thresholds=thr, action_only=acto): acc_m2 += 1
                        if compare(M3, GT, thresholds=thr, action_only=acto): acc_m3 += 1
                        
                        # print("mms_trial", mms_trial)
                        # print("str_trial", str_trial)
                        # print("L", L, "G", G)
                        all_ex += 1
       
    print(f"Number of all samples: {all_ex}")
    print(f"Acc BL: {round(100*acc_bl/all_ex,1)}%")
    print(f"Acc M1: {round(100*acc_m1/all_ex,1)}%")
    print(f"Acc M2: {round(100*acc_m2/all_ex,1)}%")
    print(f"Acc M3: {round(100*acc_m3/all_ex,1)}%")


def compare(y,ytrue, thresholds=False, action_only=False):
    """
    Args:
        y (ProbVector[3]): (Template,Selection,Storage) ProbVectors
        ytrue (Str[3]): Ground truth (Template,Selection,Storage)
        thresholds (bool, optional): If False, max() is retrieved. Defaults to False.
        action_only (bool, optional): Checking only if template matches. Defaults to False.

    Returns:
        bool: Compare match
    """    
    if thresholds:
        t = 'activated'
    else:
        t = 'max'

    if getattr(y['template'], t) == ytrue[0]:
        if action_only:
            return True
        if ytrue[1] == "" or getattr(y['selections'],t) == ytrue[1]:
            if ytrue[2] == "" or getattr(y['storages'],t) == ytrue[2]:
                return True

    # print(f"{getattr(y['template'], t)} == {ytrue[0]} and {getattr(y['selections'],t)} == {ytrue[1]} and {getattr(y['storages'],t)} == {ytrue[2]}")

    return False


if __name__ == '__main__':
    # print("=== arity ===")
    # print("All results: ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=True, ta,to,ts must match")
    # tester_all(exp=['arity'], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=True, acto=False)
    # print("Only aligned actions: ['a'], thresholding=True, ta,to,ts must match")
    # tester_all(exp=['arity'], alignm=['a'], thr=True, acto=False)
    # print("No thresholding (max): ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=False, ta,to,ts must match")
    # tester_all(exp=['arity'], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=False, acto=False)
    # print("Checking action match only: ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=True, Only ta must match")
    # tester_all(exp=['arity'], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=True, acto=True)

    print("=== property ===")
    print("All results: ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=True, ta,to,ts must match")
    tester_all(exp=['property'], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=True, acto=False)
    print("Only aligned actions: ['a'], thresholding=True, ta,to,ts must match")
    tester_all(exp=['property'], alignm=['a'], thr=True, acto=False)
    print("No thresholding (max): ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=False, ta,to,ts must match")
    tester_all(exp=['property'], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=False, acto=False)
    print("Checking action match only: ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=True, Only ta must match")
    tester_all(exp=['property'], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=True, acto=True)

