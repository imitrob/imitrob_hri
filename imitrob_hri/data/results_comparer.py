
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


class Results6DComparer():
    def __init__(self):
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
        
        
        self.data = results_all
        
        ''' Merge function index'''
        self.baseline = 0, 
        ''' baseline model is special case where:
            1. Merge function is argmax (self.baseline=0)
            2. Model is always the M1 (self.M1=0)
        '''
        self.mul = 1
        self.add_2 = 2 
        self.entropy = 3
        self.entropy_add_2 = 4
        
        ''' Metric name index '''
        self.acc_o = 0
        self.acc = 1
        self.prc = 2
        self.rcl = 3
        self.spc = 4
        self.f1  = 5
        
        self.C1 = 0
        ''' Configuration1()'s index '''
        self.C2 = 1
        ''' Configuration2()'s index '''
        self.C3 = 2
        ''' Configuration3()'s index '''
        
        self.n1 = 0
        ''' Noise level 1 refers to id=0 index '''
        self.n2 = 1
        ''' Noise level 2 refers to id=1 index '''        
        self.n3 = 2
        ''' Noise level 3 refers to id=2 index '''
        self.n4 = 3
        ''' Noise level 4 refers to id=3 index '''

        
        self.ns = slice(None,None,None) #[self.n1, self.n2, self.n3]
        ''' All Noise levels '''
        
        self.D1 = 0
        ''' Trivial '''
        self.D2 = 1
        ''' Decisible based on action properties (M2 solves that) '''
        self.D3 = 2
        ''' Decisible based on object properties (M3 solves that) '''
        self.D4 = 3
        ''' Undecisible '''
        self.D5 = 4
        ''' Only single modality information activated '''
        self.Ds = [self.D1, self.D2, self.D3, self.D4, self.D5]
        ''' All datasets '''
        self.Ds_des = [self.D1, self.D2, self.D3, self.D5]
        ''' Only decisible '''
        
        self.M1 = 0
        self.M2 = 1
        self.M3 = 2
        ''' All models '''
        self.Ms = slice(None,None,None) #[self.M1, self.M2, self.M3]


    def _1_ablation_study(self):
        ''' Here we compare Models '''
        cols = ['baseline','M1', 'M2', 'M3']
        ''' with Generated Datasets '''
        indx = ['$D1$','$D2$','$D3$','$D4$']
        
        ''' baseline model is special case where:
            1. Merge function is argmax (self.baseline=0)
            2. Model is always the M1 (self.M1=0)
        '''
        baseline = self.extract_data(indices=[self.baseline,self.acc,self.C3,self.n3,self.Ds_des,self.M1])
        other = self.extract_data(indices=[self.entropy,self.acc,self.C3,self.n3,self.Ds_des,self.Ms])
        
        print(baseline.shape)
        print(other.shape)

        ''' to percentage '''
        data = 100 * np.hstack((baseline.reshape(4,1), other))

        print(pd.DataFrame(data, columns=cols, index=indx))

        singlehistplot_customized(data, 'exp_ablation', labels=cols, xticks=indx, xlbl='Dataset generation policy', ylbl='Accuracy [%]', plot=True)

    def extract_data(self, indices):
        ''' My supercool function to extract not only slices but different point from n dim array which is impossible to do with self.data[[1,2], [1,2,3], ...], etc.
        '''
        subdata = deepcopy(self.data)
        
        for dim in range(6):
            print("subdata.shape: ", subdata.shape)
            indices_placeholder = [slice(None,None,None)] * 6
            if isinstance(indices[dim], int):
                indices[dim] = slice(indices[dim], indices[dim]+1)
            
            indices_placeholder[dim] = indices[dim]
            print("indices_placeholder: ", indices_placeholder)
            subdata = deepcopy(subdata[indices_placeholder])
        
        shp = np.array(list(subdata.shape))
        final_shape = shp[shp != 1]    
        
        print("before reshape: ", subdata.shape)
        subdata = subdata.reshape(final_shape)
        print("after reshape: ", subdata.shape)
        
        return subdata

    def _2_noise_influence(self, magic=None):
        ''' Here we compare Models '''
        cols = ['$n_1$','$n_2$', '$n_3$', '$n_4$']
        ''' with Generated Datasets '''
        indx = ['$c_2,D1$','$c_2,D2$', '$c_2,D3$', '$c_2,D4$', '$c_2,D5$'] #, '$c_3,D2$', '$c_3,D3$', '$c_3,D4$']
        
        noise_levels=self.extract_data(indices=[magic,self.acc,self.C2,self.ns,[self.D1, self.D2,self.D3,self.D4,self.D5],self.M3])
        noise_levels = noise_levels.T
        print( pd.DataFrame(100*noise_levels, columns=cols, index=indx))

        singlehistplot_customized(100*noise_levels, f'exp_noise_c2_{magic}', labels=cols, xticks=indx, xlbl='', ylbl='Accuracy [%]',bottom=0, plot=True)
        
        ''' Here we compare Models '''
        cols = ['$n_1$','$n_2$', '$n_3$', '$n_4$']
        ''' with Generated Datasets '''
        indx = ['$c_3,D1$','$c_3,D2$', '$c_3,D3$', '$c_3,D4$', '$c_3,D5$']
        
        noise_levels=self.extract_data(indices=[magic,self.acc,self.C3,self.ns,[self.D1, self.D2,self.D3,self.D4,self.D5],self.M3])
        noise_levels = noise_levels.T
        print( pd.DataFrame(100*noise_levels, columns=cols, index=indx))

        singlehistplot_customized(100*noise_levels, f'exp_noise_c3_{magic}', labels=cols, xticks=indx, xlbl='', ylbl='Accuracy [%]',bottom=0, plot=True)
        

    def _3_types_merging(self):
        ''' Here we compare generated datasets '''
        cols = ['D1', 'D2', 'D3', 'D4']
        ''' with merge fun (mul,add) and different metrics 
        (accuracy, precision,recall,specificity,f1) '''
        indx = ['$mul_{accuracy}$', '$add_{accuracy}$', '$mul_{precision}$', '$add_{precision}$', '$mul_{recall}$', '$add_{recall}$','$mul_{specificity}$', '$add_{specificity}$','$mul_{f1}$', '$add_{f1}$']

        data1=self.extract_data(indices=[[self.mul, self.add_2],self.acc,self.C3,self.n3,self.Ds_des,self.M3])
        print("Accuracy:")
        print( pd.DataFrame(100*data1, columns=cols, index=['mul', 'add']))

        data2=self.extract_data(indices=[[self.mul, self.add_2],self.prc,self.C3,self.n3,self.Ds_des,self.M3])
        print("Precision:")
        print( pd.DataFrame(100*data2, columns=cols, index=['mul', 'add']))

        data3=self.extract_data(indices=[[self.mul, self.add_2],self.rcl,self.C3,self.n3,self.Ds_des,self.M3])
        print("Recall:")
        print( pd.DataFrame(100*data3, columns=cols, index=['mul', 'add']))

        data4=self.extract_data(indices=[[self.mul, self.add_2],self.spc,self.C3,self.n3,self.Ds_des,self.M3])
        print("Specificity:")
        print( pd.DataFrame(100*data4, columns=cols, index=['mul', 'add']))

        data5=self.extract_data(indices=[[self.mul, self.add_2],self.f1,self.C3,self.n3,self.Ds_des,self.M3])
        print("F1:")
        print( pd.DataFrame(100*data5, columns=cols, index=['mul', 'add']))

        data = np.vstack((data1,data2,data3,data4,data5))

        singlehistplot_customized(100*data, 'exp_merge_methods', labels=cols, xticks=indx, xlbl='Metrics', ylbl='Accuracy [%]', plot=True)

    def _4_thresholding(self):
        ''' Here we compare Datasets '''
        cols = ['$D1$', '$D2$', '$D3$', '$D4$']
        ''' with merge functions '''
        indx = ['$baseline$','$mul_{fixed}$','$add_{fixed}$', '$mul_{entropy}$', '$add_{entropy}$']

        data = np.vstack((
        self.extract_data(indices=[self.baseline,self.acc,self.C3,self.n3,self.Ds_des,self.M1]),
        self.extract_data(indices=[[self.mul,self.add_2,self.entropy,self.entropy_add_2],self.acc,self.C3,self.n3,self.Ds_des,self.M3])
        ))

        print( pd.DataFrame(100*data, columns=cols, index=indx))

        singlehistplot_customized(100*data.T, 'exp_thresholding', labels=indx, xticks=cols, xlbl='Generation Policies', ylbl='Accuracy [%]', plot=True)

    def _5_noise_levels_compared_to_models(self):
        ''' Here we compare Models '''
        cols = ['M_1', 'M_2', 'M_3']
        ''' with Noises '''
        indx = ['n_1', 'n_2', 'n_3']

        data_accumulated = np.zeros((3,3))
        for d in self.Ds_des:
            data = self.data[self.mul,self.acc,self.C3,self.ns,d,self.Ms]
            #singlehistplot_customized(100*data.T, 'exp_thresholding', labels=index, xticks=columns, ylbl='Accuracy [%]', plot=True)
            
            data_accumulated += data
            print( pd.DataFrame(100*data, columns=cols, index=indx) )

        print( pd.DataFrame(100*data_accumulated, columns=cols, index=indx) )
        print( data_accumulated[0] )


        singlehistplot_customized((100/4)*data_accumulated.T - (100/4)*data_accumulated[0:1].T, 'exp_thresholdin_comparison', labels=indx, xticks=cols, ylbl='Accuracy [%]', plot=True)

    def _6_baseline_examination(self):
        ''' Here we compare Models '''
        cols = ['C1', 'C2', 'C3']
        ''' with Generated Datasets '''
        indx = ['$D1$','$D2$','$D3$','$D4$','$D5$']
        
        
        data = self.extract_data(indices=[self.baseline,self.acc,[self.C1,self.C2,self.C3],self.n4,self.Ds,self.M1]).T

        print(pd.DataFrame(100*data, columns=cols, index=indx))
        singlehistplot_customized(100*data, 'baseline_examination', labels=cols, xticks=indx, xlbl='Baseline examination', ylbl='Accuracy [%]', plot=True)

    def _7_some_custom_plot2(self):
        ''' Here we compare generated datasets '''
        cols = ['D1', 'D2', 'D3', 'D4']
        ''' with merge fun (mul,add) and different metrics 
        (accuracy, precision,recall,specificity,f1) '''
        indx = ['$mul_{accuracy}$', '$add_{accuracy}$', '$mul_{precision}$', '$add_{precision}$', '$mul_{recall}$', '$add_{recall}$','$mul_{specificity}$', '$add_{specificity}$','$mul_{f1}$', '$add_{f1}$']

        data1 = self.data[1:3,self.acc,self.C3,self.n3,self.Ds_des,self.M3]
        print("Accuracy:")
        print( pd.DataFrame(100*data1, columns=cols, index=['mul', 'add']))

        data2 = self.data[1:3,self.prc,self.C3,self.n3,self.Ds_des,self.M3]
        print("Precision:")
        print( pd.DataFrame(100*data2, columns=cols, index=['mul', 'add']))

        data3 = self.data[1:3,self.rcl,self.C3,self.n3,self.Ds_des,self.M3]
        print("Recall:")
        print( pd.DataFrame(100*data3, columns=cols, index=['mul', 'add']))

        data4 = self.data[1:3,self.spc,self.C3,self.n3,self.Ds_des,self.M3]
        print("Specificity:")
        print( pd.DataFrame(100*data4, columns=cols, index=['mul', 'add']))

        data5 = self.data[1:3,self.f1,self.C3,self.n3,self.Ds_des,self.M3]
        print("F1:")
        print( pd.DataFrame(100*data5, columns=cols, index=['mul', 'add']))

        data = np.vstack((data1,data2,data3,data4,data5))

        singlehistplot_customized(100*data, 'exp_merge_methods', labels=cols, xticks=indx, xlbl='Metrics', ylbl='Accuracy [%]', plot=True)


if __name__ == '__main__':
    rc = Results6DComparer()

    # rc._1_ablation_study()
    # rc._2_noise_influence(magic=rc.entropy)
    # rc._2_noise_influence(magic=rc.add_2)
    # rc._2_noise_influence(magic=rc.entropy_add_2)
    # rc._2_noise_influence(magic=rc.mul)
    
    # Results6DComparer()._3_types_merging()
    # Results6DComparer()._4_thresholding()
    # _5_noise_levels_compared_to_models()
    rc._6_baseline_examination()
    
    
def old():
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
