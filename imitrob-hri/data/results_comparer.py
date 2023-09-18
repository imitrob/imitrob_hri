
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
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
from merging_modalities.utils import singlehistplot_customized

def _1_ablation_study():
    c = 2
    n = 2
    baseline = results_all[0,1,c,n,[0,1,2,4],0]
    other =    results_all[3,1,c,n,[0,1,2,4],0:3]

    print(baseline.shape)
    print(other.shape)

    data = 100 * np.hstack((baseline.reshape(4,1), other))

    print( pd.DataFrame(100*data, columns=['baseline', 'm1', 'm2', 'm3'], index=['D1', 'D2', 'D3', 'D4']))

    singlehistplot_customized(data, 'exp_ablation', labels=['baseline','M1', 'M2', 'M3'], xticks=['$D1$','$D2$','$D3$','$D4$'], xlbl='Dataset generation policy', ylbl='Accuracy [%]', plot=True)

def _2_noise_influence():
    m = 0
    noise_levels = results_all[3,1,0,0:3,0:3,m]
    
    noise_levels = np.swapaxes(noise_levels,1,2)
    noise_levels = noise_levels.reshape(6,3)
    print(noise_levels)
    print( pd.DataFrame(100*noise_levels, columns=['n1', 'n2', 'n3'], index=['c2,D1', 'c2,D2', 'c2,D3', 'c3,D1', 'c3,D2', 'c3,D3']))

    singlehistplot_customized(100*noise_levels, 'exp_noise', labels=['$n_1$','$n_2$', '$n_3$'], xticks=['$c_2,D1$', '$c_2,D2$'] , xlbl='', ylbl='Accuracy [%]',bottom=80, plot=True)
    #'$c_2,D3$', '$c_3,D1$', '$c_3,D2$', '$c_3,D3$']

def _3_types_merging():
    c = 2
    n = 2
    m = 2

    data1 = results_all[1:3,1,c,n,[0,1,2,4],m]
    print("Accuracy:")
    print( pd.DataFrame(100*data1, columns=['D1', 'D2', 'D3', 'D4'], index=['mul', 'add']))

    data2 = results_all[1:3,2,c,n,[0,1,2,4],m]
    print("Precision:")
    print( pd.DataFrame(100*data2, columns=['D1', 'D2', 'D3', 'D4'], index=['mul', 'add']))

    data3 = results_all[1:3,3,c,n,[0,1,2,4],m]
    print("Recall:")
    print( pd.DataFrame(100*data3, columns=['D1', 'D2', 'D3', 'D4'], index=['mul', 'add']))

    data4 = results_all[1:3,4,c,n,[0,1,2,4],m]
    print("Specificity:")
    print( pd.DataFrame(100*data4, columns=['D1', 'D2', 'D3', 'D4'], index=['mul', 'add']))

    data5 = results_all[1:3,5,c,n,[0,1,2,4],m]
    print("F1:")
    print( pd.DataFrame(100*data5, columns=['D1', 'D2', 'D3', 'D4'], index=['mul', 'add']))

    data = np.vstack((data1,data2,data3,data4,data5))

    singlehistplot_customized(100*data, 'exp_merge_methods', labels=['D1', 'D2', 'D3', 'D4'], xticks=['$mul_{accuracy}$', '$add_{accuracy}$', '$mul_{precision}$', '$add_{precision}$', '$mul_{recall}$', '$add_{recall}$','$mul_{specificity}$', '$add_{specificity}$','$mul_{f1}$', '$add_{f1}$'], xlbl='Metrics', ylbl='Accuracy [%]', plot=True)

def _4_thresholding():

    c = 2
    n = 2
    m = 2

    data = np.vstack((results_all[0,1,c,n,[0,1,2,4],0], results_all[1:4,1,c,n,[0,1,2,4],m]))
    print( pd.DataFrame(100*data, columns=['D1', 'D2', 'D3', 'D4'], index=['baseline','$mul_{TH}$','$add_{TH}$', '$mul_{entropy}$']))

    singlehistplot_customized(100*data.T, 'exp_thresholding', labels=['$baseline$','$mul_{fixed}$','$add_{fixed}$', '$mul_{entropy}$'], xticks=['$D1$', '$D2$', '$D3$', '$D4$'], xlbl='Generation Policies', ylbl='Accuracy [%]', plot=True)


if __name__ == '__main__':
    # 1. Load results data into 6D table
    # (merge function) x (metric) x (config) x (noise) x (dataset) x (model)

    results_all = []

    for name in ['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']:
        accs = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/results/accs_{name}.npy", allow_pickle=True)

        results = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/results/results_{name}.npy", allow_pickle=True)
        
        results_6D = []
        results_6D.append(np.array(accs))
        results_6D.append(np.array(get_from_results('template', 'accuracy', results))) # template_accuracy
        results_6D.append(np.array(get_from_results('template', 'precision', results))) # template_precision
        results_6D.append(np.array(get_from_results('template', 'recall', results))) # template_recall
        results_6D.append(np.array(get_specificity('template', results))) # template_specificity
        results_6D.append(np.array(get_f1('template', results))) # template_f1

        results_all.append(deepcopy(results_6D))

    results_all = np.asarray(results_all)

    print(results_all.shape)

    def make_table(c, n, m, d):
        print(f"Configuration id: {c}, Noise level id: {n}, Method id: {m}, Dataset policy: {d}")
        print( pd.DataFrame(100*results_all[:,1:,c,n,d,m], columns=['accuracy', 'precision', 'recall', 'specificity', 'f1'], index=['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']))

    # make_table(c=2, n=2, m=2, p=0)
    # make_table(c=2, n=2, m=2, p=1)
    # make_table(c=2, n=2, m=2, p=2)
    # make_table(c=2, n=2, m=2, p=3)
    # make_table(c=2, n=2, m=2, p=4)


    def make_table_2(c, n, m, d):
        print(f"Configuration id: {c}, Noise level id: {n}, Method id: {m}, Regulation policy: {d}")
        data = np.vstack((results_all[0,1:,c,n,d,0], results_all[1:,1:,c,n,d,m]))
        print( pd.DataFrame(100*data, columns=['accuracy', 'precision', 'recall', 'specificity', 'f1'], index=['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']))

    # make_table_2(c=2, n=2, m=2, p=0)
    # make_table_2(c=2, n=2, m=2, p=1)
    # make_table_2(c=2, n=2, m=2, p=2)
    # make_table_2(c=2, n=2, m=2, p=3)
    # make_table_2(c=2, n=2, m=2, p=4)

    # c = 1
    # n = 1
    # m = 2
    # data = results_all[1:3,1,2,2,[0,1,2,4],0]
    # print( pd.DataFrame(100*data))

    # data = results_all[1:3,1,2,2,[0,1,2,4],1]
    # print( pd.DataFrame(100*data))

    # data = results_all[1:3,1,2,2,[0,1,2,4],2]
    # print( pd.DataFrame(100*data))

    # data = results_all[1:3,1,2,2,[0,1,2,4],2]
    # print( pd.DataFrame(100*data))

    # print("=-=")
    # data = results_all[1:3,1,1,1,[0,1,2,4],0]
    # print( pd.DataFrame(100*data))

    # data = results_all[1:3,1,1,1,[0,1,2,4],2]
    # print( pd.DataFrame(100*data))
    # data = results_all[1:3,1,1,1,[0,1,2,4],2]
    # print( pd.DataFrame(100*data))

    # print("=-=")
    # data = results_all[1:3,1,2,1,[0,1,2,4],0]
    # print( pd.DataFrame(100*data))

    # data = results_all[1:3,1,2,1,[0,1,2,4],2]
    # print( pd.DataFrame(100*data))
    # data = results_all[1:3,1,2,1,[0,1,2,4],2]
    # print( pd.DataFrame(100*data))
    # print(" ==== ")

    # c = 1
    # n = 1
    # m = 2
    # data1 = results_all[1:3,1,c,n,0:4,m]
    # print("Accuracy:")
    # print( pd.DataFrame(100*data1, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    # data2 = results_all[1:3,2,c,n,0:4,m]
    # print("Precision:")
    # print( pd.DataFrame(100*data2, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    # data3 = results_all[1:3,3,c,n,0:4,m]
    # print("Recall:")
    # print( pd.DataFrame(100*data3, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    # data4 = results_all[1:3,4,c,n,0:4,m]
    # print("Specificity:")
    # print( pd.DataFrame(100*data4, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    # data5 = results_all[1:3,5,c,n,0:4,m]
    # print("F1:")
    # print( pd.DataFrame(100*data5, columns=['p1', 'p2', 'p3', 'p4'], index=['mul', 'add']))

    _1_ablation_study()
    _2_noise_influence()
    _3_types_merging()
    _4_thresholding()