
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from nlp_new.nlp_utils import make_conjunction
import data.datagen_utils as datagen_utils 
import numpy as np

if __name__ == '__main__':
    dataset_n = int(sys.argv[1])

    dataset = np.load(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_0{dataset_n}.npy'), allow_pickle=True)
    
    ''' Set configuration '''
    acc = 0
    print(dataset)
    nsamples = len(dataset)
    for n,sample in enumerate(dataset):
        print(f"{'*' * 10} {n}th sample {'*' * 10}")
        c = sample['config']
        s = sample['x_sentence'] 
        s.make_conjunction(c)
        
        mm = ModalityMerger(c)
        s.M = mm.feedforward2(s.L, s.G, epsilon=c.epsilon, gamma=c.gamma)

        if s.check_merged(sample['y'], c):
            acc +=1
        else:
            print("y", sample['y'])

            print(s.M['template'])
            print(s.M['selections'])

    print(f"Final acc: {acc/nsamples*100}%")

