
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from nlp_new.nlp_utils import make_conjunction
import data.datagen_utils as datagen_utils 
import numpy as np

def tester_all():
    accs = np.zeros((3,3,4,3))
    for cn,c in enumerate(['c1', 'c2', 'c3']):
        for nn,n in enumerate(['n1', 'n2', 'n3']):
            for pn,p in enumerate(['p0','p1','p2','p3']):
                for mn,m in enumerate([3]):
                    dataset = np.load(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_{c}_{n}_{p}.npy'), allow_pickle=True)
                    acc = tester_on_data(dataset, m, printer=False)
                    accs[cn,nn,pn,mn] = acc
                    print(f"{c} {n} {p} {m}: {acc}")
                    np.save("/home/petr/Downloads/accs.npy", accs)
    np.save("/home/petr/Downloads/accs.npy", accs)

def tester_on_data(dataset, model, printer=False):
    ''' Set configuration '''
    acc = 0
    nsamples = len(dataset)
    for n,sample in enumerate(dataset):
        if printer: print(f"{'*' * 10} {n}th sample {'*' * 10}")
        c = sample['config']
        s = sample['x_sentence'] 
        s.make_conjunction(c)
        
        mm = ModalityMerger(c)
        s.M, DEBUGdata = mm.feedforward2(s.L, s.G, scene=sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=model)

        if s.check_merged(sample['y'], c, printer):
            acc +=1
        else:
            if printer:
                print("Scene:", sample['x_scene'])

                print("y", sample['y'])

                print(s.M['template'])
                print(s.M['selections'])
                print(s.M['storages'])
                print("-- post --")
                print(s.L['template'])
                print(s.L['selections'])
                print(s.L['storages'])
                print(s.G['template'])
                print(s.G['selections'])
                print(s.G['storages'])
                print("-- end post --")
                #print(DEBUGdata)
                #input()

    print(f"Final acc: {acc/nsamples*100}%")
    return acc/nsamples*100

if __name__ == '__main__':
    dataset_n = sys.argv[1]
    if dataset_n == 'all':
        tester_all()
    model = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    dataset = np.load(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_{dataset_n}.npy'), allow_pickle=True)
    tester_on_data(dataset, model, printer=True)
