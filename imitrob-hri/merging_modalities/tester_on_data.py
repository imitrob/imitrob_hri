
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from nlp_new.nlp_utils import make_conjunction
import data.datagen_utils as datagen_utils 
import numpy as np

if __name__ == '__main__':
    dataset_n = int(sys.argv[1])
    model = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    dataset = np.load(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_0{dataset_n}.npy'), allow_pickle=True)
    
    ''' Set configuration '''
    acc = 0
    nsamples = len(dataset)
    for n,sample in enumerate(dataset):
        print(f"{'*' * 10} {n}th sample {'*' * 10}")
        c = sample['config']
        s = sample['x_sentence'] 
        s.make_conjunction(c)
        
        mm = ModalityMerger(c)
        s.M = mm.feedforward2(s.L, s.G, scene=sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=model)

        if s.check_merged(sample['y'], c):
            acc +=1
        else:
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


    print(f"Final acc: {acc/nsamples*100}%")

