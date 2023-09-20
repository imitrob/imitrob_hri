
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from nlp_new.nlp_utils import make_conjunction
import data.datagen_utils as datagen_utils 
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from tester_on_data import tester_on_data

def tester_all(use_magic):
    accs = np.zeros((3,3,5,3))
    results_save = np.zeros((3,3,5,3), dtype=object)
    for cn,c in enumerate(['c2']):
        for nn,n in enumerate(['n2']):
            for pn,d in enumerate(['D5']):
                for mn,m in enumerate([1,2,3]):
                    red_acc = []
                    for redundancy in [1,2,3]:
                        
                        dataset = np.load(os.path.expanduser(f'{os.path.dirname(os.path.abspath(__file__))}/../data/saves/artificial_dataset_{c}_{n}_{d}.npy'), allow_pickle=True)
                        acc, results = tester_on_data(dataset, m, use_magic, printer=False)
                        red_acc.append(acc)
                        accs[cn,nn,pn,mn] = acc
                        print(f"{c} {n} {d} {m}: {acc}")
                        #print(cn,nn,pn,mn, results)
                        #results_save[cn,nn,pn,mn] = np.asanyarray(results, dtype=object)
                        #print(results)
                        #np.save(f"{os.path.dirname(os.path.abspath(__file__))}/results/accs_{use_magic}.npy", accs)
                        #np.save(f"{os.path.dirname(os.path.abspath(__file__))}/results/results_{use_magic}.npy", results_save)
                    print("Redundancy check: ", red_acc, " std: ", np.std(red_acc))
                    if np.std(red_acc) > 0.1:
                        print("Redundancy check failed!!!")
    exit()

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    use_magic = sys.argv[2]
    if dataset_name == 'all':
        tester_all(use_magic)
    model = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    dataset = np.load(os.path.expanduser(f'{os.path.dirname(os.path.abspath(__file__))}/../data/results/artificial_dataset_{dataset_name}.npy'), allow_pickle=True)
    tester_on_data(dataset, model, use_magic=use_magic, printer=True)
