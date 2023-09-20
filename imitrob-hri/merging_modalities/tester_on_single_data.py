
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from nlp_new.nlp_utils import make_conjunction
import data.datagen_utils as datagen_utils 
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

def tester_on_single_data(dataset, sample_n, model, use_magic, printer=False):
    ''' Set configuration '''
    sample = dataset[sample_n]

    if printer: print(f"{'*' * 10} {sample_n}th sample {'*' * 10}")
    c = sample['config']
    s = sample['x_sentence'] 
    s.make_conjunction(c)
    
    mm = ModalityMerger(c, use_magic)
    s.M, DEBUGdata = mm.feedforward3(s.L, s.G, scene=sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=model, use_magic=use_magic)

    if s.check_merged(sample['y'], c, printer=True):
        print("Merged!")

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

    y_true_ct, y_pred_ct = s.get_true_and_pred(sample['y'], c)
    print("y_true_ct, y_pred_ct", y_true_ct, y_pred_ct)

    category = 'template'

    cl = s.L[category].p
    cg = s.G[category].p
    non = None
    action_names = s.L[category].names
    cm = s.M[category].p
    #interactive_probs_plotter(cl, cg, action_names, c.clear_threshold, c.unsure_threshold, c.diffs_threshold, mm.mms[category])
    interactive_probs_plotter(cm, non, action_names, c.clear_threshold, c.unsure_threshold, c.diffs_threshold, mm.mms[category], save=True, save_file=f'supercoolphoto_{model}')

    return None

if __name__ == '__main__':

    dataset_name = 'c3_n3_D3'
    model = 2

    dataset = np.load(os.path.expanduser(f'{os.path.dirname(os.path.abspath(__file__))}/../data/saves/artificial_dataset_{dataset_name}.npy'), allow_pickle=True)
    use_magic = 'mul'
    sample_n = np.random.randint(0, 10000)
    for model in [1,2,3]:
        tester_on_single_data(dataset, sample_n, model, use_magic=use_magic, printer=True)
