
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score

def jakobysingleplot(data, filepartname, plot=False):
    m1 = data[:,:,:,0].reshape(36)
    m2 = data[:,:,:,1].reshape(36)
    m3 = data[:,:,:,2].reshape(36)


    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(7,5))

    # set height of bar
    #teleop =   [48, 118, 119]

    # gesture1 = [35, 98, 104]
    # gesture1_ = [16,43,56]

    # gesture2 = [34, 40, 39]
    # gesture2_ = [7, 9, 8]

    # Set position of bar on X axis
    br1 = np.arange(36)
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
            for p in ['p0','p1','p2','p3']:
                gen_x_axis_list.append(f"{c},{n},{p}")

    plt.xlabel('Setup (c - config, n - noise level, p - dataset policy)', fontsize = 15)
    plt.ylabel('Accuracy [%]', fontsize = 15)
    plt.xticks([r + barWidth for r in range(36)],
            gen_x_axis_list)
    plt.xticks(rotation=90)

    plt.legend()
    plt.savefig(f"/home/petr/Pictures/merging_modalities_{name}_{filepartname}.eps")
    plt.savefig(f"/home/petr/Pictures/merging_modalities_{name}_{filepartname}.png")
    if plot:
        plt.show()


name = sys.argv[1]
accs = np.load(f"/home/petr/Downloads/accs_{name}.npy", allow_pickle=True)
jakobysingleplot(accs, filepartname='accs', plot=True)

# ========================================================================

results = np.load(f"/home/petr/Downloads/results_{name}.npy", allow_pickle=True)



def get_from_results(ct, metric):
    ret = np.zeros((3,3,4,3))
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n1', 'n2', 'n3']):
            for pn,p in enumerate(['p0','p1','p2','p3']):
                for mn,m in enumerate(['m1', 'm2', 'm3']):
                    ret[cn,nn,pn,mn] = results[cn,nn,pn,mn].item()[ct][metric]
    return ret

def get_specificity(ct):
    ret = np.zeros((3,3,4,3))
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n1', 'n2', 'n3']):
            for pn,p in enumerate(['p0','p1','p2','p3']):
                for mn,m in enumerate(['m1', 'm2', 'm3']):
                    Y_pred = results[cn,nn,pn,mn].item()[ct]['y_pred_cts']
                    Y_true = results[cn,nn,pn,mn].item()[ct]['y_true_cts']

                    cm = multilabel_confusion_matrix(Y_true, Y_pred)
                    specificity = []
                    for item in cm:
                        tn, fp, fn, tp = item[0,0], item[0,1], item[1,0], item[1,1]
                        specificity.append( tn / (tn + fp) )
                    specificity = np.array(specificity).mean()

                    ret[cn,nn,pn,mn] = specificity
    return ret

def get_f1(ct):
    ret = np.zeros((3,3,4,3))
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n1', 'n2', 'n3']):
            for pn,p in enumerate(['p0','p1','p2','p3']):
                for mn,m in enumerate(['m1', 'm2', 'm3']):
                    Y_pred = results[cn,nn,pn,mn].item()[ct]['y_pred_cts']
                    Y_true = results[cn,nn,pn,mn].item()[ct]['y_true_cts']
                    ret[cn,nn,pn,mn] = f1_score(Y_true, Y_pred, average='micro')
    return ret

template_accuracy = get_from_results('template', 'accuracy')
template_precision = get_from_results('template', 'precision')
template_recall = get_from_results('template', 'recall')

template_specificity = get_specificity('template')
template_f1 = get_f1('template')


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


