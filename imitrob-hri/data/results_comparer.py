
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from extraction_funs import *
from copy import deepcopy
import pandas as pd
accs_6D = []
results_all = []

for name in ['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']:
    accs = np.load(f"/home/petr/Downloads/accs_{name}.npy", allow_pickle=True)
    accs_6D.append(accs)

    results = np.load(f"/home/petr/Downloads/results_{name}.npy", allow_pickle=True)
    
    results_6D = []
    results_6D.append(np.array(accs))
    results_6D.append(np.array(get_from_results('template', 'accuracy', results))) # template_accuracy
    results_6D.append(np.array(get_from_results('template', 'precision', results))) # template_precision
    results_6D.append(np.array(get_from_results('template', 'recall', results))) # template_recall
    results_6D.append(np.array(get_specificity('template', results))) # template_specificity
    results_6D.append(np.array(get_f1('template', results))) # template_f1

    results_all.append(deepcopy(results_6D))


accs_6D = np.asarray(accs_6D)
results_all = np.asarray(results_all)

## It's a large dimensional table, but it should have everything, so I can pick whatever I want
#accs[magic][c][n][p][m]
accs_6D_ = accs_6D[:,:,:,:,:]

#print(accs_6D_.shape)

# results all: magic x metric x c x n x p x m
print(results_all.shape)

## Addition vs Multiplication?


def make_table(c, n, m, p):
    print(f"Configuration id: {c}, Noise level id: {n}, Method id: {m}, Regulation policy: {p}")
    print( pd.DataFrame(100*results_all[:,1:,c,n,p,m], columns=['accuracy', 'precision', 'recall', 'specificity', 'f1'], index=['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']))


## Regulation P4:
# make_table(c=2, n=2, m=2, p=0)
# make_table(c=2, n=2, m=2, p=1)
# make_table(c=2, n=2, m=2, p=2)
# make_table(c=2, n=2, m=2, p=3)
# make_table(c=2, n=2, m=2, p=4)


def make_table_2(c, n, m, p):
    print(f"Configuration id: {c}, Noise level id: {n}, Method id: {m}, Regulation policy: {p}")

    tt = np.vstack((results_all[0,1:,c,n,p,0], results_all[1:,1:,c,n,p,m]))


    print( pd.DataFrame(100*tt, columns=['accuracy', 'precision', 'recall', 'specificity', 'f1'], index=['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']))

if False:
    make_table_2(c=2, n=2, m=2, p=0)
    make_table_2(c=2, n=2, m=2, p=1)
    make_table_2(c=2, n=2, m=2, p=2)
    make_table_2(c=2, n=2, m=2, p=3)
    make_table_2(c=2, n=2, m=2, p=4)


#np.save("/home/petr/Downloads/results_magic_metric_c_n_p_m.npy", results_all)

def _1_ablation_study():

    c = 2
    n = 2
    #                ->.
    #           baseline ->.,M ,
    baseline = results_all[0,1,c,n,:,0]
    #entropy mul.
    other =    results_all[3,1,c,n,:,0:3]

    print(baseline.shape)
    print(other.shape)

    tt = np.hstack((baseline.reshape(5,1), other))


    print( pd.DataFrame(100*tt, columns=['baseline', 'm1', 'm2', 'm3'], index=['p1', 'p2', 'p3', 'p4', 'p5']))


#_1_ablation_study()

def _2_noise_influence():

    m = 2
    c = 2
    noise_levels = results_all[1,1,1:3,0:3,0:3,m]

    noise_levels = np.swapaxes(noise_levels,1,2)
    noise_levels = noise_levels.reshape(6,3)
    print(noise_levels)
    print( pd.DataFrame(100*noise_levels, columns=['n1', 'n2', 'n3'], index=['c2  p1', 'c2  p2', 'c2  p3', 'c3  p1', 'c3  p2', 'c3  p3']))


#_2_noise_influence()

def _3_types_merging():
    c = 1
    n = 1
    m = 2

    data = results_all[1:3,1,c,n,0:4,m]
    print("Accuracy:")
    print( pd.DataFrame(100*data, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    data = results_all[1:3,2,c,n,0:4,m]
    print("Precision:")
    print( pd.DataFrame(100*data, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    data = results_all[1:3,3,c,n,0:4,m]
    print("Recall:")
    print( pd.DataFrame(100*data, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    data = results_all[1:3,4,c,n,0:4,m]
    print("Specificity:")
    print( pd.DataFrame(100*data, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    data = results_all[1:3,5,c,n,0:4,m]
    print("F1:")
    print( pd.DataFrame(100*data, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

_3_types_merging()

def _4_thresholding():

    c = 1
    n = 1
    m = 2

    tt = np.vstack((results_all[0,1:,c,n,p,0], results_all[1:,1:,c,n,p,m]))

    data = results_all[0:3,1,c,n,0:4,m]
    print("Accuracy:")
    print( pd.DataFrame(100*data, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))