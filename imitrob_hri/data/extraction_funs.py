
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix


def get_from_results(ct, metric, results):
    ret = np.zeros((3,7,4,3))
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']):
            for pn,p in enumerate(['D1','D2','D3','D4']):
                for mn,m in enumerate(['m1', 'm2', 'm3']):
                    ret[cn,nn,pn,mn] = results[cn,nn,pn,mn].item()[ct][metric]
    return ret

def get_specificity(ct, results):
    ret = np.zeros((3,7,4,3))
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']):
            for pn,p in enumerate(['D1','D2','D3','D4']):
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

def get_f1(ct, results):
    ret = np.zeros((3,7,4,3))
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']):
            for pn,p in enumerate(['D1','D2','D3','D4']):
                for mn,m in enumerate(['m1', 'm2', 'm3']):
                    Y_pred = results[cn,nn,pn,mn].item()[ct]['y_pred_cts']
                    Y_true = results[cn,nn,pn,mn].item()[ct]['y_true_cts']
                    ret[cn,nn,pn,mn] = f1_score(Y_true, Y_pred, average='micro')
    return ret
