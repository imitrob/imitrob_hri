
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from imitrob_nlp.nlp_utils import make_conjunction
import data.datagen_utils as datagen_utils 
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

def tester_all(use_magic):
    accs = np.zeros((3,3,5,3))
    results_save = np.zeros((3,3,5,3), dtype=object)
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n1', 'n2', 'n3']):
            for pn,d in enumerate(['D1','D2','D3','D4','D5']):
                for mn,m in enumerate(['M1','M2','M3']):
                    dataset = np.load(os.path.expanduser(f'{os.path.dirname(os.path.abspath(__file__))}/../data/saves/artificial_dataset_{c}_{n}_{d}.npy'), allow_pickle=True)
                    model = int(m[1]) # (from M1 chooses the 1)
                    acc, results = tester_on_data(dataset, model, use_magic, printer=False)
                    accs[cn,nn,pn,mn] = acc
                    print(f"{c} {n} {d} {m}: {acc}")
                    #print(cn,nn,pn,mn, results)
                    results_save[cn,nn,pn,mn] = np.asanyarray(results, dtype=object)
                    #print(results)
                    np.save(f"{os.path.dirname(os.path.abspath(__file__))}/../data/results/accs_{use_magic}.npy", accs)
                    np.save(f"{os.path.dirname(os.path.abspath(__file__))}/../data/results/results_{use_magic}.npy", results_save)
    

def tester_on_data(dataset, model, use_magic, printer=False):
    ''' Set configuration '''
    y_pred_cts = []
    y_true_cts = []

    acc = 0
    nsamples = len(dataset)
    for n,sample in enumerate(dataset):
        if n > 1000: break
        if printer: print(f"{'*' * 10} {n}th sample {'*' * 10}")
        c = sample['config']
        s = sample['x_sentence'] 
        s.make_conjunction(c)
        
        mm = ModalityMerger(c, use_magic)
        s.M, DEBUGdata = mm.feedforward3(s.L, s.G, scene=sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=model, use_magic=use_magic)

        if s.check_merged(sample['y'], c, printer):
            acc +=1
        else:
            if printer:
                print("-- Scene: --")
                print(sample['x_scene'])
                print("-- Input Language --")
                print(s.L['template'])
                print(s.L['selections'])
                print(s.L['storages'])
                print("-- Input Gesture --")
                print(s.G['template'])
                print(s.G['selections'])
                print(s.G['storages'])
                print("-- Output --")
                print(s.M['template'])
                print(s.M['selections'])
                print(s.M['storages'])
                print("-- True Value: --")                
                print(sample['y'])

                #print(DEBUGdata)
                #input()
        y_true_ct, y_pred_ct = s.get_true_and_pred(sample['y'], c)
        y_pred_cts.append(y_pred_ct)
        y_true_cts.append(y_true_ct)

    y_pred_cts = np.asarray(y_pred_cts)
    y_true_cts = np.asarray(y_true_cts)

    results = {}
    for ct,ctn in enumerate(['template', 'selections', 'storages']): # todo
        results[ctn] = {
        'precision': precision_score(y_true_cts[:,ct], y_pred_cts[:,ct], average='micro'),
        'recall': recall_score(y_true_cts[:,ct], y_pred_cts[:,ct], average='micro'),
        'accuracy': accuracy_score(y_true_cts[:,ct], y_pred_cts[:,ct]),
        'y_true_cts': y_true_cts[:,ct],
        'y_pred_cts': y_pred_cts[:,ct],
        }

    if printer: print(f"Final acc: {acc/n*100}%")
    return acc/n*100, results

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    if dataset_name == 'all':
        if len(sys.argv) > 2:
            use_magic = sys.argv[2]
            print(f"Running on single merge function: {use_magic}")
            tester_all(use_magic)
        else:
            print(f"Running on all merge functions!")
            for mgn,use_magic in enumerate(['mul', 'add_2', 'entropy', 'entropy_add_2']):
                tester_all(use_magic)
    else:
        model = sys.argv[3] if len(sys.argv) > 3 else 'M3'

        dataset = np.load(os.path.expanduser(f'{os.path.dirname(os.path.abspath(__file__))}/../data/results/artificial_dataset_{dataset_name}.npy'), allow_pickle=True)
        use_magic = sys.argv[2]
        tester_on_data(dataset, model, use_magic=use_magic, printer=True)
