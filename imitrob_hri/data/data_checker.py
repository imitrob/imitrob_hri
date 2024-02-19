import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import numpy as np
from collections import Counter

from imitrob_hri.merging_modalities.modality_merger import ModalityMerger

DATASET_PREFIX = '2' # '' for original dataset

def check_single(dataset_name, all=True):

    dataset = np.load(os.path.expanduser(f'{os.path.dirname(os.path.abspath(__file__))}/saves/artificial_dataset{DATASET_PREFIX}_{dataset_name}.npy'), allow_pickle=True)
    print(f"Dataset: {dataset_name}")

    tmplts = []
    slctns = []
    strgs = []
    for sample in dataset:
        tmplts.append( sample['y']['template'] )
        slctns.append( sample['y']['selections'] )
        strgs.append( sample['y']['storages'] )

    print(f"Target actions: {dict(Counter(tmplts))}")
    print(f"Target objects: {dict(Counter(slctns))}")
    print(f"Target storages: {dict(Counter(strgs))}")

    print("--- Property True/False occurances ---")

    glued_n = 0
    pickable_n = 0
    reachable_n = 0
    stackable_n = 0
    pushable_n = 0
    full_n = 0
    full_liquid_n = 0

    for sample in dataset:
        glued_n += int(sample['x_scene'].selections[0].glued()[1])
        pickable_n += int(sample['x_scene'].selections[0].pickable()[1])
        reachable_n += int(sample['x_scene'].selections[0].reachable()[1])
        stackable_n += int(sample['x_scene'].selections[0].stackable()[1])
        pushable_n += int(sample['x_scene'].selections[0].pushable()[1])
        full_n += int(sample['x_scene'].selections[0].full_stack()[1])
        full_liquid_n += int(sample['x_scene'].selections[0].full_liquid()[1])

    TO_PERCENT = 100 / len(dataset)
    print(f"Property: glued, pickable, reachable, stackable, pushable, full, full_liquid")
    #print(glued_n, pickable_n, reachable_n, stackable_n, pushable_n, full_n, full_liquid_n)
    print(f"True %:{round(glued_n*TO_PERCENT,1)}%, {round(pickable_n*TO_PERCENT,1)}%, {round(reachable_n*TO_PERCENT,1)}%, {round(stackable_n*TO_PERCENT,1)}%, {round(pushable_n*TO_PERCENT,1)}%, {round(full_n*TO_PERCENT,1)}%, {round(full_liquid_n*TO_PERCENT,1)}%")
    print("---")

    if all:
        for n,sample in enumerate(dataset):
            print(f"-------------- SAMPLE {n} --------------")
            
            print(f"y: {sample['y']['template']}, {sample['y']['selections']}, {sample['y']['storages']}")
            print(f"config {sample['config']}")
            print(f"x {sample['x_scene']}")
            print(f"x {sample['x_sentence']}")

            print("target_action probs")
            # print(sample['x_sentence'].L['template'].max)
            print(sample['x_sentence'].L['template'].activated)
            print(sample['x_sentence'].L['selections'].activated)
            print(sample['x_sentence'].L['storages'].activated)

            c = dataset[n]['config']
            
            s = sample['x_sentence'] 
            # s.make_conjunction(c)

            print("--- Merged M1 ---")
            print("[[[[[[MUL]]]]]]")
            mm = ModalityMerger(c, "mul")
            s.make_conjunction(c)
            s.M, DEBUGdata = mm.feedforward3(s.L, s.G, sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=1, use_magic="mul")
            print(s.M['template'])
            print(s.M['selections'])
            print(s.M['storages'])
            print("[[[[[[ENTROPY]]]]]]")
            mm = ModalityMerger(c, "entropy")
            s.make_conjunction(c)
            
            s.M, DEBUGdata = mm.feedforward3(s.L, s.G, sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=1, use_magic="entropy")
            print(s.M['template'])
            print(s.M['selections'])
            print(s.M['storages'])
            
            mm = ModalityMerger(c, "entropy")
            s.make_conjunction(c)
            
            s.M, DEBUGdata = mm.feedforward3(s.L, s.G, sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=3, use_magic="entropy")

            print("--- Merged M3 ---")
            print(s.M['template'])
            print(s.M['selections'])
            print(s.M['storages'])
            print(DEBUGdata)

            input("next sample?")

    input("")


def check_all():
    for c in ['c1', 'c2', 'c3']:
        for n in ['n1', 'n2', 'n3']:
            for D in ['D1','D2','D3','D4','D5']:
                check_single(f"{c}_{n}_{D}", all=False)

if __name__ == '__main__':
    ''' Loads the dataset and prints some info:
    1. Which target_actions, object selections, storages are there?
    2. Properties on/off
    '''
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = input("Enter dataset name as: cX_nX_DX, e.g. 'c1_n1_D1': ")
    
    if dataset_name == 'all':
        check_all()
    else:
        check_single(dataset_name, all=True)