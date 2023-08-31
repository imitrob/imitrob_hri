from configuration import Configuration
from modality_merger import ProbsVector, SingleTypeModalityMerger, ModalityMerger, MultiProbsVector
from utils import *
import sys, os; sys.path.append("..")
from nlp_new.nlp_utils import make_conjunction, to_default_name
import numpy as np

if __name__ == '__main__':
    dataset = np.load(os.path.expanduser('~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_04.npy'), allow_pickle=True)
    
    ''' Set configuration '''
    c = dataset[0]['config']

    acc = 0
    nsamples = len(dataset)
    for n,sample in enumerate(dataset):
        print(f"{'*' * 10} {n}th sample {'*' * 10}")
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

