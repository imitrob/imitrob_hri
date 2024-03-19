
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

DATASET_PREFIX = '2' # '' for original dataset

DATASET_DATE = '' # '_26_2' # '' for up to date

class Results6DComparer():
    def __init__(self):
        # 1. Load results data into 6D table
        # (merge function) x (metric) x (config) x (noise) x (dataset) x (model)

        results_all = []

        for name in ['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']:
            accs = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/results{DATASET_PREFIX}{DATASET_DATE}/accs_{name}.npy", allow_pickle=True)

            results = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/results{DATASET_PREFIX}{DATASET_DATE}/results_{name}.npy", allow_pickle=True)
            
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
        
        self.merge_fun_to_str = [
            '$baseline$', '$mul_{fixed}$', '$add_{fixed}$', '$mul_{entropy}$', '$add_{entropy}$'
        ]
        self.merge_fun_to_str_noeq = [
            'baseline', 'mul_fixed', 'add_fixed', 'mul_entropy', 'add_entropy'
        ]
        ''' Merge function index'''
        self.baseline = 0
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
        
        self.n0 = 0
        ''' Noise level 1 refers to id=0 index '''
        self.n1 = 1
        ''' Noise level 2 refers to id=1 index '''        
        self.n2 = 2
        ''' Noise level 3 refers to id=2 index '''
        self.n3 = 3
        ''' Noise level 4 refers to id=3 index '''
        self.n4 = 4
        self.n5 = 5
        self.n6 = 6

        self.n0_new = self.n0
        self.n1_new = self.n3
        self.n2_new = self.n1
        self.n3_new = self.n2
        self.n4_new = self.n6
        
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
        # self.D5 = 4
        ''' Only single modality information activated '''
        self.Ds = [self.D1, self.D2, self.D3, self.D4]
        ''' All datasets '''
        self.Ds_des = [self.D1, self.D2, self.D3, self.D4]
        ''' Only decisible '''
        
        self.M1 = 0
        self.M2 = 1
        self.M3 = 2
        ''' All models '''
        self.Ms = slice(None,None,None) #[self.M1, self.M2, self.M3]

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

    def _1_ablation_study(self, magic=None):
        ''' Here we compare Models '''
        cols = ['baseline','M1', 'M2', 'M3']
        ''' with Generated Datasets '''
        indx = ['$D1$','$D2$','$D3$','$D4$']
        
        ''' baseline model is special case where:
            1. Merge function is argmax (self.baseline=0)
            2. Model is always the M1 (self.M1=0)
        '''
        baseline = self.extract_data(indices=[self.baseline,self.acc,self.C3,self.n0,self.Ds_des,self.M1])
        other = self.extract_data(indices=[magic,self.acc,self.C3,self.n0,self.Ds_des,self.Ms])
        
        print(baseline.shape)
        print(other.shape)

        ''' to percentage '''
        data = 100 * np.hstack((baseline.reshape(4,1), other))

        print(pd.DataFrame(data, columns=cols, index=indx))

        singlehistplot_customized(data, f'exp_ablation_{magic}', labels=cols, xticks=indx, xlbl='Dataset generation policy', ylbl='Accuracy [%]', plot=True, title=f'Ablation study: $C_3$, $n_0$, {self.merge_fun_to_str[magic]}')


    def _2_noise_influence(self, magic=None):
        ''' Here we compare Models '''
        cols = ['$n_0$','$n_1$', '$n_2$', '$n_3$','$n_4$','$n_5$']
        ''' with Generated Datasets '''
        indx = ['$c_2,D1$','$c_2,D2$', '$c_2,D3$', '$c_2,D4$'] #, '$c_3,D2$', '$c_3,D3$', '$c_3,D4$']
        
        noise_levels=self.extract_data(indices=[magic,self.acc,self.C2,self.ns,[self.D1, self.D2,self.D3,self.D4],self.M3])
        noise_levels = noise_levels.T
        print( pd.DataFrame(100*noise_levels, columns=cols, index=indx))

        singlehistplot_customized(100*noise_levels, f'exp_noise_c2_{magic}', labels=cols, xticks=indx, xlbl='', ylbl='Accuracy [%]',bottom=0, plot=True, title=f'Noise levels: $M_3$, $C_2$, {self.merge_fun_to_str[magic]}')
        
        ''' Here we compare Models '''
        cols = ['$n_0$','$n_1$', '$n_2$', '$n_3$','$n_4$','$n_5$']
        ''' with Generated Datasets '''
        indx = ['$c_3,D1$','$c_3,D2$', '$c_3,D3$', '$c_3,D4$']
        
        noise_levels=self.extract_data(indices=[magic,self.acc,self.C3,self.ns,[self.D1, self.D2,self.D3,self.D4],self.M3])
        noise_levels = noise_levels.T
        print( pd.DataFrame(100*noise_levels, columns=cols, index=indx))

        singlehistplot_customized(100*noise_levels, f'exp_noise_c3_{magic}', labels=cols, xticks=indx, xlbl='', ylbl='Accuracy [%]',bottom=0, plot=True, title=f'Noise levels: $M_3$, $C_3$, {self.merge_fun_to_str[magic]}')


    def _3_types_merging(self):
        ''' Here we compare generated datasets '''
        cols = ['D1', 'D2', 'D3', 'D4']
        ''' with merge fun (mul,add) and different metrics 
        (accuracy, precision,recall,specificity,f1) '''
        indx = ['$mul_{accuracy}$', '$add_{accuracy}$', '$mul_{precision}$', '$add_{precision}$', '$mul_{recall}$', '$add_{recall}$','$mul_{specificity}$', '$add_{specificity}$','$mul_{f1}$', '$add_{f1}$']

        data1=self.extract_data(indices=[[self.mul, self.add_2],self.acc,self.C3,self.n5,self.Ds_des,self.M3])
        print("Accuracy:")
        print( pd.DataFrame(100*data1, columns=cols, index=['mul', 'add']))

        data2=self.extract_data(indices=[[self.mul, self.add_2],self.prc,self.C3,self.n5,self.Ds_des,self.M3])
        print("Precision:")
        print( pd.DataFrame(100*data2, columns=cols, index=['mul', 'add']))

        data3=self.extract_data(indices=[[self.mul, self.add_2],self.rcl,self.C3,self.n5,self.Ds_des,self.M3])
        print("Recall:")
        print( pd.DataFrame(100*data3, columns=cols, index=['mul', 'add']))

        data4=self.extract_data(indices=[[self.mul, self.add_2],self.spc,self.C3,self.n5,self.Ds_des,self.M3])
        print("Specificity:")
        print( pd.DataFrame(100*data4, columns=cols, index=['mul', 'add']))

        data5=self.extract_data(indices=[[self.mul, self.add_2],self.f1,self.C3,self.n5,self.Ds_des,self.M3])
        print("F1:")
        print( pd.DataFrame(100*data5, columns=cols, index=['mul', 'add']))

        data = np.vstack((data1,data2,data3,data4,data5))

        singlehistplot_customized(100*data, 'exp_merge_methods', labels=cols, xticks=indx, xlbl='Metrics', ylbl='Accuracy [%]', plot=True, title="Merging types: $M_3$, $n_5$, $C_3$")

    def _4_thresholding(self, n, M):
        ''' Here we compare Datasets '''
        cols = ['$D1$', '$D2$', '$D3$', '$D4$']
        ''' with merge functions '''
        indx = ['$baseline$','$mul_{fixed}$','$add_{fixed}$', '$mul_{entropy}$', '$add_{entropy}$']

        data = np.vstack((
        self.extract_data(indices=[self.baseline,self.acc,self.C3,n,self.Ds_des,self.M1]),
        self.extract_data(indices=[[self.mul,self.add_2,self.entropy,self.entropy_add_2],self.acc,self.C3,n,self.Ds_des,M])
        ))

        print( pd.DataFrame(100*data, columns=cols, index=indx))

        singlehistplot_customized(100*data.T, f'exp_thresholding_n{n}_M{M+1}', labels=indx, xticks=cols, xlbl='Generation Policies', ylbl='Accuracy [%]', plot=True, title=f"Thresholding: $M_{M+1}$, $n_{n}$, $C_3$")

    # OLD PLOT 4
    def plot_4_thresholding_final(self, n, M):
        ''' Here we compare Datasets '''
        cols = ['$D_{align}^{sim}$', '$D_{arity}^{sim}$', '$D_{property}^{sim}$', '$D_{unaligned}^{sim}$', '$D_{align}^{real}$', '$D_{unnaligned}^{real}$']
        ''' with merge functions '''
        indx = ['$baseline$', '$max$', '$add_{fixed}$', '$mul_{fixed}$', '$add_{entropy}$', '$mul_{entropy}$']

        data_sim = np.vstack((
        self.extract_data(indices=[self.baseline,self.acc,self.C3,n,self.Ds_des,self.M1]),
        self.extract_data(indices=[[self.baseline, self.add_2, self.mul, self.entropy_add_2, self.entropy],self.acc,self.C3,n,self.Ds_des,M])
        )).T

        # data_real = [
        #     [0.516, 0.9835, 1, 1],
        #     [0.092, 0.867, 0.934, 0.921],
        # ]

        # data = np.vstack((data_sim, data_real)).T

        print( pd.DataFrame(100*data_sim, columns=cols, index=indx))

        singlehistplot_customized(100*data_sim.T, f'Plot_4_thresholding_final_C3_n{n}old_M{M+1}', labels=indx, xticks=cols, xlbl='Generation Policies', ylbl='Accuracy [%]', plot=True, title=f"Thresholding")

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


        singlehistplot_customized((100/4)*data_accumulated.T - (100/4)*data_accumulated[0:1].T, 'exp_thresholdin_comparison', labels=indx, xticks=cols, ylbl='Accuracy [%]', plot=True, title="Noise levels compared: $mul_{fixed}$, $C_3$")

    def _6_baseline_examination(self):
        ''' Here we compare Models '''
        cols = ['C1', 'C2', 'C3']
        ''' with Generated Datasets '''
        indx = ['$D1$','$D2$','$D3$','$D4$']
        
        
        data = self.extract_data(indices=[self.baseline,self.acc,[self.C1,self.C2,self.C3],self.n3,self.Ds,self.M1]).T

        print(pd.DataFrame(100*data, columns=cols, index=indx))
        singlehistplot_customized(100*data, 'baseline_examination', labels=cols, xticks=indx, xlbl='Baseline examination', ylbl='Accuracy [%]', plot=True, title="Baseline examination: $baseline$, $n_3$, $C_1$")

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

    def Plot_1_Ablation(self, magic=None, noise=None, noise_name=None):
        ''' Here we compare Models '''
        cols = ['$Baseline$','$M_1$', '$M_2$', '$M_3$']
        ''' with Generated Datasets '''
        indx = ['$D^{sim}_{\mathcal{A}}$','$D^{sim}_{\mathcal{U}}$', '$D^{real}_{\mathcal{A}}$', '$D^{real}_{\mathcal{U}}$']

        data_real = [
            [0.458, 0.9005, 0.9005, 1],
            [0.05, 0.3586, 0.455, 0.921]
        ]

        DS = [self.D1, self.D2, self.D3]

        baseline = self.extract_data(indices=[self.baseline,self.acc,self.C3,noise,DS,self.M1])
        other = self.extract_data(indices=[magic,self.acc,self.C3,noise,DS,self.Ms])

        print(baseline.shape)
        print(other.shape)

        ''' JOIN D2 and D3 '''
        sim_data = np.hstack((baseline.reshape(len(DS),1), other))
        # 3 Datasets x 4 Models
        assert sim_data.shape == (3, 4)
        # Join D2 and D3 
        D2D3_average = (sim_data[1] + sim_data[2]) / 2
        sim_data = sim_data[0:2]
        sim_data[1] = D2D3_average

        data = 100 * np.vstack((sim_data, data_real))

        print(pd.DataFrame(data, columns=cols, index=indx))

        singlehistplot_customized(data, f'Plot_1_Ablation_{self.merge_fun_to_str_noeq[magic]}_C3_{noise_name}', labels=cols, xticks=indx, xlbl='Datasets', ylbl='Accuracy [%]', plot=False, title=f'', legend_loc='lower left', figsize=(6,2.8))

    def Plot_2_Noises_Models(self, merge_fun):
        ''' Here we compare Datasets '''
        cols = ['$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$',
                '$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$',
                '$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$',
                '$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$'] #, '$D_{\mathcal{U}}^{sim}$'] #, '$D_{align}^{real}$', '$D_{unnaligned}^{real}$']
        ''' with merge functions '''
        indx = ['$n_0$','$n_1^{real}$', '$n_2$', '$n_3$','$n_4$']

        DS = [self.D1, self.D2, self.D3]

        # 4 modely - aligned x notaligned (only for sim) for all noises
        # Noises are in legend
        # Models and datasets(unaligned, aligned) are columns;;; D^A, D^U -> M_1

        noises_new = [self.n0_new,self.n1_new,self.n2_new,self.n3_new,self.n4_new]
        
        data_pieces = []
        for n,m in enumerate([self.baseline, self.M1, self.M2, self.M3]):
            if n == 0:
                data_piece = self.extract_data(indices=[self.baseline,self.acc,self.C3,noises_new,DS,m])
            else:
                data_piece = self.extract_data(indices=[merge_fun,self.acc,self.C3,noises_new,DS,m])

            # 3 Datasets x 4 Models
            assert data_piece.shape == (5, 3)
            # Join D2 and D3 
            D2D3_average = (data_piece[:,1] + data_piece[:,2]) / 2
            data_piece = data_piece[:,0:2]
            data_piece[:,1] = D2D3_average

            data_pieces.append(
                data_piece
            )
        data_sim = np.hstack(data_pieces)

        # data_real = [
        #     [0.516, 0.9835, 1, 1],
        #     [0.092, 0.867, 0.934, 0.921],
        # ]

        # data = np.vstack((data_sim, data_real)).T


        

        print( pd.DataFrame(100*data_sim, columns=cols, index=indx))

        singlehistplot_customized(100*data_sim.T, f'Plot_2_Noises_Models_C3_M3_{self.merge_fun_to_str_noeq[merge_fun]}', labels=indx, xticks=cols, xlbl='Datasets', ylbl='Accuracy [%]', plot=False, title=f"", double_xticks = True)

    def Plot_3_Noise_Mergefun(self):
        ''' Here we compare Models '''
        ''' with Generated Datasets '''
        cols = ['$Baseline$','$max$', '$mul$', '$add$']
        indx = ['$n_1^{real}$', '$n_2$', '$n_3$','$n_4$']
        
        noises = [self.n1_new,self.n2_new,self.n3_new,self.n4_new]

        data_sim = np.vstack((
        self.extract_data(indices=[self.baseline,self.acc,self.C3,noises,self.D1,self.M1]),
        self.extract_data(indices=[[self.baseline, self.entropy, self.entropy_add_2],self.acc,self.C3,noises,self.D1,self.M3])
        )).T
        
        print( pd.DataFrame(100*data_sim, columns=cols, index=indx))

        singlehistplot_customized(100*data_sim, f'Plot_3_Noise_Mergefun_c3_D1_M3', labels=cols, xticks=indx, xlbl='Noise levels', ylbl='Accuracy [%]',bottom=0, plot=False, title=f'', figsize=(3,2.3))

    def Plot_4_Noise_Mergefun_Alt(self):
        ''' Here we compare Models '''
        ''' with Generated Datasets '''
        indx = ['$Baseline$','$max$', '$mul$', '$add$']
        cols = ['$n_0$','$n_1^{real}$', '$n_2$', '$n_3$','$n_4$']
        
        noises = [self.n0_new,self.n1_new,self.n2_new,self.n3_new,self.n4_new]

        data_sim = np.vstack((
        self.extract_data(indices=[self.baseline,self.acc,self.C3,noises,self.D1,self.M1]),
        self.extract_data(indices=[[self.baseline, self.entropy, self.entropy_add_2],self.acc,self.C3,noises,self.D1,self.M3])
        ))
        
        print( pd.DataFrame(100*data_sim, columns=cols, index=indx))

        singlehistplot_customized(100*data_sim, f'Plot_4_Noise_Mergefun_Alt_c3_D1_M3', labels=cols, xticks=indx, xlbl='Merge function', ylbl='Accuracy [%]',bottom=0, plot=False, title=f'', legend_loc='lower right', disable_y_ticks=True, figsize=(3,2.3))

    def Plot_5_Thresholding(self, n, M, noise_name):
        ''' Here we compare Datasets '''
        cols = ['$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$', '$D_{\mathcal{A}}^{real}$', '$D_{\mathcal{U}}^{real}$']
        ''' with merge functions '''
        indx = ['$Baseline$', '$max$', '$mul_{fixed}$','$mul_{entropy}$', '$add_{fixed}$', '$add_{entropy}$']

        DS = [self.D1, self.D2, self.D3]

        data_sim = np.vstack((
        self.extract_data(indices=[self.baseline,self.acc,self.C3,n,DS,self.M1]),
        self.extract_data(indices=[[self.baseline, self.mul, self.entropy, self.add_2, self.entropy_add_2],self.acc,self.C3,n,DS,M])
        )).T

        
        ''' Join D2 and D3 ''' 
        # 3 Datasets x 4 Models
        assert data_sim.shape == (3, 6)
        # Join D2 and D3 
        D2D3_average = (data_sim[1] + data_sim[2]) / 2
        data_sim = data_sim[0:2]
        data_sim[1] = D2D3_average

        # datareal: thresholding, all_match, M3
        # baseline, max, mul, add
        data_real = [
            [0.516, 0.9835, (0.968+0.967)/2, (1+1)/2, 1, 1],
            [0.092, 0.867, (0.925+0.867)/2, (0.958+0.883)/2, 0.921, 0.934],
        ]


        data = np.vstack((data_sim, data_real)).T

        print( pd.DataFrame(100*data, columns=cols, index=indx))

        singlehistplot_customized(100*data.T, f'Plot_5_Thresholding_C3_{noise_name}_M{M+1}', labels=indx, xticks=cols, xlbl='Datasets', ylbl='Accuracy [%]', plot=False, title=f"", figsize=(6,2.3))


    def Last_Day_Special(self):
        ''' Here we compare Datasets '''
        cols = ['$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$', '$D_{\mathcal{A}}^{real}$', '$D_{\mathcal{U}}^{real}$']
        ''' with merge functions '''
        indx = ['$add_{\text{No thr.}}$', '$add_{fixed}$','$add_{entropy}$']



        data_n3 = np.array([
            [86.7, 85.3, 84.2],
            [88.4, 86.1, 83.6],
            [100, 100, 100],
            [93.4, 92.1, 93.4],
        ]).T
        data_n1 = np.array([
            [98.9, 98.4, 98.4],
            [98.7, 98.3,  97.2],
            [100, 100, 100],
            [93.4, 92.1,  93.4],
        ]).T

        
        print( pd.DataFrame(data_n3, columns=cols, index=indx))

        singlehistplot_customized(data_n3.T, f'ThresholdingSpecial_C3_n3_M3', labels=indx, xticks=cols, xlbl='Datasets', ylbl='Accuracy [%]', plot=False, title=f"", figsize=(6,2.3))

        print( pd.DataFrame(data_n1, columns=cols, index=indx))

        singlehistplot_customized(data_n1.T, f'ThresholdingSpecial_C3_n1_M3', labels=indx, xticks=cols, xlbl='Datasets', ylbl='Accuracy [%]', plot=False, title=f"", figsize=(6,2.3))

    def Last_Day_Special_f1(self):
        cols = ['$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$', '$D_{\mathcal{A}}^{real}$', '$D_{\mathcal{U}}^{real}$']
        indx = ['$add_{\text{No thr.}}$', '$add_{fixed}$','$add_{entropy}$']
        # ORIGINAL
        # cols = ['$D_{\mathcal{A}}^{sim}$', '$D_{\mathcal{U}}^{sim}$', '$D_{\mathcal{A}}^{real}$', '$D_{\mathcal{U}}^{real}$']
        # indx = ['$Baseline$', '$max$', '$mul_{fixed}$','$mul_{entropy}$', '$add_{fixed}$', '$add_{entropy}$']

        DS = [self.D1, self.D2, self.D3]

        data_sim = np.vstack((
        # self.extract_data(indices=[self.baseline,self.f1,self.C3,self.n1_new,DS,self.M1]),
        self.extract_data(indices=[[self.add_2, self.entropy_add_2],self.f1,self.C3,self.n1_new,DS,self.M3])
        )).T
        ''' Join D2 and D3 ''' 
        D2D3_average = (data_sim[1] + data_sim[2]) / 2
        data_sim = data_sim[0:2]
        data_sim[1] = D2D3_average

        data_sim_n3 = np.vstack((
        # self.extract_data(indices=[self.baseline,self.f1,self.C3,self.n1_new,DS,self.M1]),
        self.extract_data(indices=[[self.add_2, self.entropy_add_2],self.f1,self.C3,self.n3_new,DS,self.M3])
        )).T
        ''' Join D2 and D3 ''' 
        D2D3_average = (data_sim_n3[1] + data_sim_n3[2]) / 2
        data_sim_n3 = data_sim_n3[0:2]
        data_sim_n3[1] = D2D3_average


        data_n1 = np.array([
            # No thr, fixed, entropy
            [98.4, None, None],  # SIM  AL
            [98.3, None,  None], # SIM  UN

            [100, 96.8, 100],     # REAL AL
            [94.2, 93.3,  94.2], # REAL UN
        ])
        data_n3 = np.array([
            # No thr, fixed, entropy
            [85.3, None, None],  # SIM  AL
            [(82.6+98.4)/2, None,  None], # SIM  UN

            [100, 96.8, 100],     # REAL AL
            [94.2, 93.3,  94.2], # REAL UN
        ])
        print(data_n1)
        data_n1[0:2,1:3] = 100*data_sim
        data_n3[0:2,1:3] = 100*data_sim_n3
        print(data_n1)

        data_n1 = data_n1.T
        data_n3 = data_n3.T

        print( pd.DataFrame(data_n1, columns=cols, index=indx))
        singlehistplot_customized(data_n1.T, f'ThresholdingSpecial_f1_C3_n1_M3', labels=indx, xticks=cols, xlbl='Datasets', ylbl='F1 [%]', plot=False, title=f"", figsize=(6,2.3))
        print( pd.DataFrame(data_n3, columns=cols, index=indx))
        singlehistplot_customized(data_n3.T, f'ThresholdingSpecial_f1_C3_n3_M3', labels=indx, xticks=cols, xlbl='Datasets', ylbl='F1 [%]', plot=False, title=f"", figsize=(6,2.3))


if __name__ == '__main__':
    rc = Results6DComparer()

    # rc.Plot_1_Ablation(magic=rc.entropy, noise=rc.n0_new, noise_name='n0')
    # rc.Plot_1_Ablation(magic=rc.entropy, noise=rc.n1_new, noise_name='n1')
    # rc.Plot_1_Ablation(magic=rc.entropy, noise=rc.n2_new, noise_name='n2')
    # rc.Plot_1_Ablation(magic=rc.entropy, noise=rc.n3_new, noise_name='n3')
    # rc.Plot_1_Ablation(magic=rc.entropy, noise=rc.n4_new, noise_name='n4')

    # rc.Plot_1_Ablation(magic=rc.entropy_add_2, noise=rc.n0_new, noise_name='n0')
    # FINAL
    #rc.Plot_1_Ablation(magic=rc.entropy_add_2, noise=rc.n1_new, noise_name='n1')
    # rc.Plot_1_Ablation(magic=rc.entropy_add_2, noise=rc.n2_new, noise_name='n2')
    # rc.Plot_1_Ablation(magic=rc.entropy_add_2, noise=rc.n3_new, noise_name='n3')
    # rc.Plot_1_Ablation(magic=rc.entropy_add_2, noise=rc.n4_new, noise_name='n4')


    # rc.Plot_2_Noises_Models(merge_fun = rc.entropy)
    # rc.Plot_2_Noises_Models(merge_fun = rc.entropy_add_2)

    # rc.Plot_3_Noise_Mergefun()

    # rc.Plot_4_Noise_Mergefun_Alt()

    # rc.Plot_5_Thresholding(n=rc.n0_new, M=rc.M3, noise_name='n0')
    # rc.Plot_5_Thresholding(n=rc.n1_new, M=rc.M3, noise_name='n1')
    # rc.Plot_5_Thresholding(n=rc.n2_new, M=rc.M3, noise_name='n2')
    # rc.Plot_5_Thresholding(n=rc.n3_new, M=rc.M3, noise_name='n3')
    # rc.Plot_5_Thresholding(n=rc.n4_new, M=rc.M3, noise_name='n4')

    # rc.Last_Day_Special()
    rc.Last_Day_Special_f1()

def old_plots():
    
    rc.plot_4_thresholding_final(n=rc.n2_new, M=rc.M3)


    rc._1_ablation_study(magic=rc.entropy)
    rc._1_ablation_study(magic=rc.add_2)
    rc._1_ablation_study(magic=rc.entropy_add_2)
    rc._1_ablation_study(magic=rc.mul)

    rc._2_noise_influence(magic=rc.entropy)
    rc._2_noise_influence(magic=rc.add_2)
    rc._2_noise_influence(magic=rc.entropy_add_2)
    rc._2_noise_influence(magic=rc.mul)
    
    rc._3_types_merging()
    
    rc._4_thresholding(n=rc.n4, M=rc.M2)
    rc._4_thresholding(n=rc.n5, M=rc.M2)
    rc._4_thresholding(n=rc.n4, M=rc.M3)
    rc._4_thresholding(n=rc.n5, M=rc.M3)

    rc._4_thresholding_final(n=rc.n0_new, M=rc.M3)
    rc._4_thresholding_final(n=rc.n1_new, M=rc.M3)
    rc._4_thresholding_final(n=rc.n2_new, M=rc.M3)
    rc._4_thresholding_final(n=rc.n3_new, M=rc.M3)
    rc._4_thresholding_final(n=rc.n4_new, M=rc.M3)

    rc._5_noise_levels_compared_to_models()
    
    rc._6_baseline_examination()
    
    
def super_old_plots():
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
