
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

def jakobysingleplot(data, filepartname, plot=False):
    m1 = data[:,:,:,0].reshape(45)
    m2 = data[:,:,:,1].reshape(45)
    m3 = data[:,:,:,2].reshape(45)


    # set width of bar
    barWidth = 0.25
    fig = plt.figure(figsize =(14,8))

    # set height of bar
    #teleop =   [48, 118, 119]

    # gesture1 = [35, 98, 104]
    # gesture1_ = [16,43,56]

    # gesture2 = [34, 40, 39]
    # gesture2_ = [7, 9, 8]

    # Set position of bar on X axis
    br1 = np.arange(45)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.grid(axis='y')
    # Make the plot
    #plt.bar(br1, teleop, color ='r', width = barWidth,
    #        edgecolor ='grey', label ='Tele-operation')

    colors = iter([plt.cm.tab20(i) for i in range(20)])

    plt.bar(br3, m3, color =next(colors), width = barWidth,
            edgecolor ='grey', label ='M3')
    plt.bar(br2, m2, color =next(colors), width = barWidth,
            edgecolor ='grey', label ='M2')
    plt.bar(br1, m1, color =next(colors), width = barWidth,
            edgecolor ='grey', label ='M1')


    # plt.bar(br3, gesture2, color ='lightblue', width = barWidth,
    #         edgecolor ='grey', label ='Hi-lvl ActGs \nRunning')
    # plt.bar(br3, gesture2_, color ='b', width = barWidth,
    #         edgecolor ='grey', label ='Hi-lvl ActGs \nUser Performs')


    # Adding Xticks
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    gen_x_axis_list = []
    for c in ['c1', 'c2', 'c3']:
        for n in ['n1', 'n2', 'n3']:
            for p in ['p0','p1','p2','p3','p4']:
                gen_x_axis_list.append(f"{c},{n},{p}")

    plt.xlabel('Setup (c - config, n - noise level, p - dataset policy)', fontsize = 15)
    plt.ylabel('Accuracy [%]', fontsize = 15)
    plt.xticks([r + barWidth for r in range(45)],
            gen_x_axis_list)
    plt.xticks(rotation=90)

    plt.legend()
    
    try:
        os.mkdir(f"/home/petr/Pictures/mm_pics/{name}")
    except OSError as error:
        print(error)  
    plt.savefig(f"/home/petr/Pictures/mm_pics/{name}/{filepartname}.eps", dpi=fig.dpi, bbox_inches='tight')
    plt.savefig(f"/home/petr/Pictures/mm_pics/{name}/{filepartname}.png", dpi=fig.dpi, bbox_inches='tight')
    if plot:
        plt.show()


name = sys.argv[1]
accs = np.load(f"/home/petr/Downloads/accs_{name}.npy", allow_pickle=True)
jakobysingleplot(accs, filepartname='accs', plot=True)

# ========================================================================

results = np.load(f"/home/petr/Downloads/results_{name}.npy", allow_pickle=True)

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

jakobysingleplot(template_accuracy, "template_accuracy")
jakobysingleplot(template_precision, "template_precision")
jakobysingleplot(template_recall, "template_recall")
jakobysingleplot(template_specificity, "template_specificity")
jakobysingleplot(template_f1, "template_f1")
# jakobysingleplot(selections_accuracy, "selections_accuracy")
# jakobysingleplot(selections_precision, "selections_precision")
# jakobysingleplot(selections_recall, "selections_recall")
# jakobysingleplot(storages_accuracy, "storages_accuracy")
# jakobysingleplot(storages_precision, "storages_precision")
# jakobysingleplot(storages_recall, "storages_recall")


