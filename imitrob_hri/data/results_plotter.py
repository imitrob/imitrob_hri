
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from extraction_funs import *

def singlehistplot(data, filepartname, plot=False):
    ''' Plot histogram across all configurations c,D,m
    
    '''
    m1 = data[:,:,:,0].reshape(45)
    m2 = data[:,:,:,1].reshape(45)
    m3 = data[:,:,:,2].reshape(45)

    # set width of bar
    barWidth = 0.25
    fig = plt.figure(figsize =(14,8))

    # Set position of bar on X axis
    br1 = np.arange(45)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.grid(axis='y')
    colors = iter([plt.cm.tab20(i) for i in range(20)])

    plt.bar(br3, m3, color =next(colors), width = barWidth,
            edgecolor ='grey', label ='M3')
    plt.bar(br2, m2, color =next(colors), width = barWidth,
            edgecolor ='grey', label ='M2')
    plt.bar(br1, m1, color =next(colors), width = barWidth,
            edgecolor ='grey', label ='M1')

    # Adding Xticks
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    gen_x_axis_list = []
    for c in ['c1', 'c2', 'c3']:
        for n in ['n1', 'n2', 'n3', 'n4']:
            for p in ['D1','D2','D3','D4','D5']:
                gen_x_axis_list.append(f"{c},{n},{p}")

    plt.xlabel('Setup (c - config, n - noise level, D - dataset policy)', fontsize = 15)
    plt.ylabel('Accuracy [%]', fontsize = 15)
    plt.xticks([r + barWidth for r in range(45)],
            gen_x_axis_list)
    plt.xticks(rotation=90)

    plt.legend()
    
    p = Path(f"{os.path.dirname(os.path.abspath(__file__))}/pictures/{name}")
    p.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(p.joinpath(f"{filepartname}.eps"), dpi=fig.dpi, bbox_inches='tight')
    plt.savefig(p.joinpath(f"{filepartname}.png"), dpi=fig.dpi, bbox_inches='tight')
    if plot:
        plt.show()

if __name__ == '__main__':

    name = sys.argv[1]
    accs = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/results/accs_{name}.npy", allow_pickle=True)
    singlehistplot(accs, filepartname='accs', plot=True)

    # ========================================================================

    results = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/results/results_{name}.npy", allow_pickle=True)

    template_accuracy = get_from_results('template', 'accuracy', results)
    template_precision = get_from_results('template', 'precision', results)
    template_recall = get_from_results('template', 'recall', results)

    template_specificity = get_specificity('template', results)
    template_f1 = get_f1('template', results)


    # selections_accuracy = get_from_results('selections', 'accuracy')
    # selections_precision = get_from_results('selections', 'precision')
    # selections_recall = get_from_results('selections', 'recall')

    # storages_accuracy = get_from_results('storages', 'accuracy')
    # storages_precision = get_from_results('storages', 'precision')
    # storages_recall = get_from_results('storages', 'recall')

    # template = results[:,:,:,0,'template','y_true_cts']
    # selections = results[:,:,:,0,'selections','y_true_cts']
    # storages = results[:,:,:,0,'storages','y_true_cts']

    # template = results[:,:,:,0,'template','y_pred_cts']
    # selections = results[:,:,:,0,'selections','y_pred_cts']
    # storages = results[:,:,:,0,'storages','y_pred_cts']

    singlehistplot(template_accuracy, "template_accuracy")
    singlehistplot(template_precision, "template_precision")
    singlehistplot(template_recall, "template_recall")
    singlehistplot(template_specificity, "template_specificity")
    singlehistplot(template_f1, "template_f1")
    # singlehistplot(selections_accuracy, "selections_accuracy")
    # singlehistplot(selections_precision, "selections_precision")
    # singlehistplot(selections_recall, "selections_recall")
    # singlehistplot(storages_accuracy, "storages_accuracy")
    # singlehistplot(storages_precision, "storages_precision")
    # singlehistplot(storages_recall, "storages_recall")


