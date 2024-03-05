
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from imitrob_nlp.nlp_utils import make_conjunction
import data.datagen_utils2 as datagen_utils2 
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path



def tester_all(use_magic='entropy_add_2'):

    data_real_folder = Path(__file__).parent.parent.joinpath('data_real')

    with open(data_real_folder.joinpath('dataset20240226_233209_res.log')) as f:
        y_file = f.read()
    y_file_lines = y_file.split("\n")
    
    ground_truth = []
    for y_file_line in y_file_lines:
        y_line_elements = y_file_line.split(",")
        ground_truth.append(y_line_elements)



    all_ex = 0
    acc_bl = 0
    acc_m1 = 0
    acc_m2 = 0
    acc_m3 = 0
    for cn,c in enumerate(['arity']):
        for pn,d in enumerate(['a', 'nag', 'nal', 'nog', 'nol']):
            
            for tn,t in enumerate([1,2]): # trial 
                for sn,s in enumerate(['r', 'k', 'p']):

                    fld = data_real_folder.joinpath(c,d)
                    for file in fld.glob(f"{c}_{d}_{t}_{s}_mms_trial_*"): # all trials
                        n_of_trial = str(file)[:-4].split("_")[-1]

                        mms_trial = np.load(fld.joinpath(f"{c}_{d}_{t}_{s}_mms_trial_{n_of_trial}.npy"), allow_pickle=True)
                        str_trial = np.load(fld.joinpath(f"{c}_{d}_{t}_{s}_str_trial_{n_of_trial}.npy"), allow_pickle=True)


                        L = mms_trial[0].L
                        G = mms_trial[0].G
                        GT = ground_truth[int(t)]
                        BL = mms_trial[0].M
                        M1 = mms_trial[1].M
                        M2 = mms_trial[2].M
                        M3 = mms_trial[3].M

                        if compare(BL, GT): acc_bl += 1 
                        if compare(M1, GT): acc_m1 += 1
                        if compare(M2, GT): acc_m2 += 1
                        if compare(M3, GT): acc_m3 += 1
                        
                        # print("mms_trial", mms_trial)
                        # print("str_trial", str_trial)
                        # print("L", L, "G", G)
                        all_ex += 1
       
    print(f"all ex: {all_ex}")
    print(f"acc bt: {acc_bl/all_ex}")
    print(f"acc m1: {acc_m1/all_ex}")
    print(f"acc m2: {acc_m2/all_ex}")
    print(f"acc m3: {acc_m3/all_ex}")


def compare(y,ytrue):
    if y['template'].max == ytrue[0]:
        if ytrue[1] == "" or y['selections'].max == ytrue[1]:
            if ytrue[2] == "" or y['storages'].max == ytrue[2]:
                return True
    
    print(f"{y['template'].activated} == {ytrue[0]} and {y['selections'].activated} == {ytrue[1]} and {y['storages'].activated} == {ytrue[2]}")
    return False


def tester_on_data(dataset, model, use_magic, printer=False):
    ''' Set configuration '''
    y_pred_cts = []
    y_true_cts = []

    test_actionsdoability_hist = []

    acc = 0
    nsamples = len(dataset)
    for n,sample in enumerate(dataset):
        if printer: print(f"{'*' * 10} {n}th sample {'*' * 10}")
        c = sample['config']
        s = sample['x_sentence'] 
        s.make_conjunction(c)
        
        mm = ModalityMerger(c, use_magic)
        s.M, DEBUGdata = mm.feedforward3(s.L, s.G, scene=sample['x_scene'], epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=model, use_magic=use_magic)

        n_actions_possible = 0
        for template in c.templates:
            tmpl = None
            for t in sample['x_scene'].templates:
                if t.name == template:
                    tmpl = t
            assert tmpl is not None
            
            # [template].is_feasible()
            isfeasible = datagen_utils2.is_action_is_feasible_given_this_scene(tmpl, sample['x_scene'])
            # print(f"template {template} is doable? {isfeasible}")
            if isfeasible:
                n_actions_possible += 1

        test_actionsdoability_hist.append(n_actions_possible)


        if s.check_merged(sample['y'], c, printer):
            acc +=1
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

                print(DEBUGdata)
                input()
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

                print(DEBUGdata)
                input()
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
    n+=1
    print(f"Final acc: {acc/n*100}%")

    # plt.hist(test_actionsdoability_hist, bins=np.linspace(0, 10, 20))
    # plt.grid()
    # plt.xlabel("Number of feasible actions on scene")
    # plt.savefig("/home/petr/Downloads/test_actionsdoability_hist.png")
    # plt.show()

    return acc/n*100, results

if __name__ == '__main__':
    tester_all()
    # dataset_name = sys.argv[1]
    # if dataset_name == 'all':
    #     if len(sys.argv) > 2:
    #         use_magic = sys.argv[2]
    #         print(f"Running on single merge function: {use_magic}")
    #         
    #     else:
    #         print(f"Running on all merge functions!")
    #         for mgn,use_magic in enumerate(['baseline', 'mul', 'add_2', 'entropy', 'entropy_add_2']):
    #             tester_all(use_magic)
    # else:
    #     # Model M3: model = 3
    #     model = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    #     printer = eval(sys.argv[4]) if len(sys.argv) > 4 else False
        

    #     dataset = np.load(os.path.expanduser(f'{os.path.dirname(os.path.abspath(__file__))}/../data/saves/artificial_dataset{DATASET_PREFIX}_{dataset_name}.npy'), allow_pickle=True)
    #     use_magic = sys.argv[2]
    #     tester_on_data(dataset, model, use_magic=use_magic, printer=printer)
