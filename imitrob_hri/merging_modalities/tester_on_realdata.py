
from copy import deepcopy
import sys, os
from imitrob_hri.data.scene3_def import create_scene_from_fake_data
from imitrob_hri.imitrob_nlp.TemplateFactory import TemplateFactory
from imitrob_hri.merging_modalities.configuration import ConfigurationCrow1
from imitrob_hri.merging_modalities.modality_merger import MMSentence; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from modality_merger import ModalityMerger
from utils import *
from imitrob_nlp.nlp_utils import make_conjunction
import data.datagen_utils2 as datagen_utils2 
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
from imitrob_hri.merging_modalities.utils import singlehistplot_customized

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
# from devtools import debug as print




def tester_all(exp=['arity'], 
               alignm=['a', 'nag', 'nal', 'nog', 'nol'],
               thr=True,
               acto=False,
               one_by_one=False,
               postprocessing=False,
               mf=None, 
               ):

    data_real_folder = Path(__file__).parent.parent.joinpath('data_real')

    with open(data_real_folder.joinpath('dataset20240226_233209_res.log')) as f:
        y_file = f.read()
    y_file_lines = y_file.split("\n")
    
    ground_truth = []
    # ground_truth = trial (log line = some trial) x (experiment,n trial,template,selection,storage)
    for y_file_line in y_file_lines:
        y_line_elements = y_file_line.split(",")
        ground_truth.append(y_line_elements)

    all_ex = 0
    acc_bl = 0
    acc_m1 = 0
    acc_m2 = 0
    acc_m3 = 0
    acc_m1_fixed = 0
    acc_m2_fixed = 0
    acc_m3_fixed = 0
    acc_bl_m3 = 0 # Max function with thresholding, added last
    
    # This is added for special evaluating purpose
    special_pred_bl = []
    special_real_bl = []
    special_pred_thr = []
    special_real_thr = []
    special_pred_fixed = []
    special_real_fixed = []
    special_pred_M3 = []
    special_real_M3 = []

    for cn,c in enumerate(exp):
        for pn,d in enumerate(alignm):
            
            for tn,t in enumerate([1,2,3,4,5,6,7,8,9,10]): # Run 
                for sn,s in enumerate(['r', 'k', 'p']):

                    fld = data_real_folder.joinpath(c,d)
                    for file in fld.glob(f"{c}_{d}_{t}_{s}_mms_trial_*"): # all trials
                        n_of_trial = str(file)[:-4].split("_")[-1]

                        mms_trial = np.load(fld.joinpath(f"{c}_{d}_{t}_{s}_mms_trial_{n_of_trial}.npy"), allow_pickle=True)
                        str_trial = np.load(fld.joinpath(f"{c}_{d}_{t}_{s}_str_trial_{n_of_trial}.npy"), allow_pickle=True)


                        L = mms_trial[0].L
                        L['template'].template_names = []
                        L['template'].p = []
                        L['selections'].template_names = []
                        L['selections'].p = []
                        L['storages'].template_names = []
                        L['storages'].p = []

                        G = mms_trial[0].G

                        print(len(L['template'].names), len(G['template'].names))
                        # search for a line, which corresponds to given experiment and trial
                        GT = []
                        for gt_line in ground_truth:
                            # print(f"{gt_line[0]} == {c}, {gt_line[0] == c} and {gt_line[1]} == {t}, {str(gt_line[1]) == str(t)}")
                            if str(gt_line[0]) == str(c) and str(gt_line[1]) == str(t):
                                GT = gt_line[2:]
                                break
                        # check gt_line found
                        assert len(GT)>0, f"line {c}, {t} not found in dataset20240226_233209_res.log"

                        # BL = mms_trial[0].M
                        # M1 = mms_trial[1].M
                        # M2 = mms_trial[2].M
                        # M3 = mms_trial[3].M
                        
                        scene = create_scene_from_fake_data(c, t)
                        # Newly generated
                        BL = mm_run_wrapper(L, G, scene, model=1, merge_fun='baseline')
                        M1 = mm_run_wrapper(L, G, scene, model=1, merge_fun=mf[0])
                        M2 = mm_run_wrapper(L, G, scene, model=2, merge_fun=mf[0])
                        M3 = mm_run_wrapper(L, G, scene, model=3, merge_fun=mf[0])
                        M1_fixed = mm_run_wrapper(L, G, scene, model=1, merge_fun=mf[1])
                        M2_fixed = mm_run_wrapper(L, G, scene, model=2, merge_fun=mf[1])
                        M3_fixed = mm_run_wrapper(L, G, scene, model=3, merge_fun=mf[1])
                        BL_M3 = mm_run_wrapper(L, G, scene, model=3, merge_fun='baseline')


                        BL_suc = compare(BL, GT, thresholds=False, action_only=acto, one_by_one=one_by_one, m='BL', scene=scene, postprocessing=postprocessing)
                        M1_suc = compare(M1, GT, thresholds=thr, action_only=acto, one_by_one=one_by_one, m='M1', scene=scene, postprocessing=postprocessing)
                        M2_suc = compare(M2, GT, thresholds=thr, action_only=acto, one_by_one=one_by_one, m='M2', scene=scene, postprocessing=postprocessing)
                        M3_suc = compare(M3, GT, thresholds=thr, 
                        action_only=acto, one_by_one=one_by_one, m='M3', scene=scene, postprocessing=postprocessing)
                        M1_suc_fixed = compare(M1_fixed, GT, thresholds=thr, action_only=acto, one_by_one=one_by_one, m='M1', scene=scene, postprocessing=postprocessing)
                        M2_suc_fixed = compare(M2_fixed, GT, thresholds=thr, action_only=acto, one_by_one=one_by_one, m='M2', scene=scene, postprocessing=postprocessing)
                        M3_suc_fixed = compare(M3_fixed, GT, thresholds=thr, 
                        action_only=acto, one_by_one=one_by_one, m='M3', scene=scene, postprocessing=postprocessing)

                        special_pred_bl.append(f"{BL['template'].activated}")#{M3['selections'].activated}{M3['storages'].activated}")
                        special_real_bl.append(f"{GT[0]}")#{GT[1]}{GT[2]}")
                        special_pred_thr.append(f"{M3['template'].activated}")#{M3['selections'].activated}{M3['storages'].activated}")
                        special_real_thr.append(f"{GT[0]}")#{GT[1]}{GT[2]}")
                        special_pred_fixed.append(f"{M3_fixed['template'].activated}")#{M3['selections'].activated}{M3['storages'].activated}")
                        special_real_fixed.append(f"{GT[0]}")#{GT[1]}{GT[2]}")

                        special_pred_M3.append(f"{BL_M3['template'].activated}")#{M3['selections'].activated}{M3['storages'].activated}")
                        special_real_M3.append(f"{GT[0]}")#{GT[1]}{GT[2]}")


                        # Baseline with M3 - added on top
                        BL_M3_suc_fixed = compare(BL_M3, GT, thresholds=False, 
                        action_only=acto, one_by_one=one_by_one, m='M3', scene=scene, postprocessing=postprocessing)

                        if BL_suc: acc_bl += 1 
                        if M1_suc: acc_m1 += 1
                        if M2_suc: acc_m2 += 1
                        if M3_suc: acc_m3 += 1
                        if M1_suc_fixed: acc_m1_fixed += 1
                        if M2_suc_fixed: acc_m2_fixed += 1
                        if M3_suc_fixed: acc_m3_fixed += 1
                        if BL_M3_suc_fixed: acc_bl_m3 += 1
                        all_ex += 1

                        if one_by_one:
                            if M1_suc and not M2_suc:
                                print(f"exp: {c}, run: {t}")
                                print(f"== VSTUP==\nL {L['template']} {L['selections']} {L['storages']}, \nG {G['template']} {G['selections']} {G['storages']}")
                                
                                print(f"== VYSTUP ==\nM1 {M1['template']} {M1['selections']}, {M1['storages']} \nM2 {M2['template']} {M2['selections']} {M2['storages']}")
                                
                                # print("mms_trial", mms_trial)
                                # print("str_trial", str_trial)
                                # print("L", L, "G", G)
                                # input(f"INSERT SETUP! (for trial number {c} {t})")

                                # scene = create_scene_from_fake_data()
                                # conf = ConfigurationCrow1()
                                # mms = MMSentence(L=L, G=G)
                                # mms.make_conjunction(conf)
                                # mm = ModalityMerger(conf, MERGE_FUN)
                                # mms.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=scene, epsilon=conf.epsilon, gamma=conf.gamma, alpha_penal=conf.alpha_penal, model=MODEL, use_magic=MERGE_FUN)
                                # print("== SCENE == ")
                                # print(scene)
                                # print(f"== MM NEW MERGE ==\nM3 {mms.M['template']} {mms.M['selections']} {mms.M['storages']}")

                                input("?? NEW TRIAL ??")
       
    # print(f"Number of all samples: {all_ex}")
    ret = [
        round(100*acc_bl/all_ex,1),
        round(100*acc_m1/all_ex,1),
        round(100*acc_m2/all_ex,1),
        round(100*acc_m3/all_ex,1),
        round(100*acc_m1_fixed/all_ex,1),
        round(100*acc_m2_fixed/all_ex,1),
        round(100*acc_m3_fixed/all_ex,1)
    ]
    # print(f"Acc BL: {ret[0]}%")
    # print(f"Acc M1: {ret[1]}%")
    # print(f"Acc M2: {ret[2]}%")
    print(f"Acc M3: {ret[3]}%")
    # print(f"Acc M1 fixed: {ret[4]}%")
    # print(f"Acc M2 fixed: {ret[5]}%")
    print(f"Acc M3 fixed: {ret[6]}%")
    # print(f"Acc BL M3: {round(100*acc_bl_m3/all_ex,1)}%")


    # evaluating special metric
    if False:
        f1_final_1 = f1_score(special_pred_bl, special_real_bl, average='micro')
        print(f"f1_final bl: {round(100*f1_final_1,1)}")
        f1_final_2 = f1_score(special_pred_thr, special_real_thr, average='micro')
        print(f"f1_final thr: {round(100*f1_final_2,1)}")
        f1_final_3 = f1_score(special_pred_fixed, special_real_fixed, average='micro')
        print(f"f1_final fixed: {round(100*f1_final_3,1)}")

        return f1_final_1, f1_final_2, f1_final_3
        # cm = confusion_matrix(special_pred_M3, special_real_M3)                        
        # fp = cm.sum(axis=0) - np.diag(cm)
        # fp_final = sum(fp) / all_ex
        # print(f"f1_final M3: {round(100*fp_final,1)}")

    # evaluating special metric
    if False:
        cm = confusion_matrix(special_pred_bl, special_real_bl)                        
        fp = cm.sum(axis=0) - np.diag(cm)
        fp_final = sum(fp) / all_ex
        print(f"fp_final bl: {round(100*fp_final,1)}")
        cm = confusion_matrix(special_pred_thr, special_real_thr)                        
        fp = cm.sum(axis=0) - np.diag(cm)
        fp_final = sum(fp) / all_ex
        print(f"fp_final thr: {round(100*fp_final,1)}")
        cm = confusion_matrix(special_pred_fixed, special_real_fixed)                        
        fp = cm.sum(axis=0) - np.diag(cm)
        fp_final = sum(fp) / all_ex
        print(f"fp_final fixed: {round(100*fp_final,1)}")
        cm = confusion_matrix(special_pred_M3, special_real_M3)                        
        fp = cm.sum(axis=0) - np.diag(cm)
        fp_final = sum(fp) / all_ex
        print(f"fp_final M3: {round(100*fp_final,1)}")


    # print(f"Acc BL list: {acc_bl_list}")
    

    return ret



def mm_run_wrapper(L, G, scene, model, merge_fun):
    
    conf = ConfigurationCrow1()
    mms = MMSentence(L=L, G=G)
    mms.make_conjunction(conf)
    mm = ModalityMerger(conf, merge_fun)
    mms.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=scene, epsilon=conf.epsilon, gamma=conf.gamma, alpha_penal=conf.alpha_penal, model=model, use_magic=merge_fun)
    return mms.M

def compare(y,ytrue, thresholds=False, action_only=False, one_by_one=False, m='BL', scene=None, postprocessing=False):
    """
    Args:
        y (ProbVector[3]): (Template,Selection,Storage) ProbVectors
        ytrue (Str[3]): Ground truth (Template,Selection,Storage)
        thresholds (bool, optional): If False, max() is retrieved. Defaults to False.
        action_only (bool, optional): Checking only if template matches. Defaults to False.

    Returns:
        bool: Compare match
    """    
    if thresholds:
        t = 'activated'
    else:
        t = 'max'


    if one_by_one:
        print(f"{getattr(y['template'], t)} == {ytrue[0]} and {getattr(y['selections'],t)} == {ytrue[1]} and {getattr(y['storages'],t)} == {ytrue[2]}")

    if not postprocessing:
        if getattr(y['template'], t) == ytrue[0]:
            if action_only:
                return True
            if ytrue[1] == "" or getattr(y['selections'],t) == ytrue[1]:
                if ytrue[2] == "" or getattr(y['storages'],t) == ytrue[2]:
                    return True
        return False
    else: # Postprocessing
        if m != 'M3':
            if getattr(y['template'], t) == ytrue[0]:
                if action_only:
                    return True
                if ytrue[1] == "" or getattr(y['selections'],t) == ytrue[1]:
                    if ytrue[2] == "" or getattr(y['storages'],t) == ytrue[2]:
                        return True
            return False
        else: # Final check if template is feasible on the scene
            # print("getattr(y['template'], t)", getattr(y['template'], t))
            template_sorted_names = []
            template_vector = deepcopy(y['template'])
            template_sorted_names.append(template_vector.max)
            if template_sorted_names[0] is None:
                return False
            # print("template sorted naems", template_sorted_names)
            # if template_sorted_names[0] != getattr(y['template'], t):
            #     print("template sorted naems", template_sorted_names)
            #     input(f"!!! ")
            
            # template_vector.pop(template_vector.max_id)
            # template_sorted_names.append(template_vector.max)
            # template_vector.pop(template_vector.max_id)
            # template_sorted_names.append(template_vector.max)
            # print("template_sorted_names", template_sorted_names)
            
            # print("template_sorted_names", template_sorted_names)
            for template_name in template_sorted_names:
                template_o = TemplateFactory().get_template_class_from_str(template_name)()

                combs = []
                combs_p = []
                if template_o.mm_pars_compulsary == ['template', 'selections', 'storages']:
                    for o in scene.selections:
                        for s in scene.storages:
                            if template_o.is_feasible(o,s):
                                combs.append((o.name, s.name))
                                combs_p.append((y['selections'].prob_for_entity(o.name) + y['storages'].prob_for_entity(s.name)) / 2)
                    # print("combs", combs, combs_p)
                    max_id = np.array(combs_p).argmax()
                    o_name, s_name = combs[max_id]
                elif template_o.mm_pars_compulsary == ['template', 'selections']:
                    for o in scene.selections:
                        if template_o.is_feasible(o,s=None):
                            combs.append(o.name)
                            combs_p.append(y['selections'].prob_for_entity(o.name))
                    # print("combs", combs, combs_p)
                    max_id = np.array(combs_p).argmax()
                    o_name = combs[max_id]
                elif template_o.mm_pars_compulsary == ['template']:
                    template_name

                
                # final check
                # if template_name != getattr(y['template'], t):
                #     print("template_name", template_name, "getattr(y['template'], t)", getattr(y['template'], t))
                #     input("???")
                if template_name == ytrue[0]:
                    if action_only:
                        return True
                    if ytrue[1] == "" or o_name == ytrue[1]:
                        if ytrue[2] == "" or s_name == ytrue[2]:
                            return True
                        else:
                            input(f"{template_name} == {ytrue[0]}, {o_name} == {ytrue[1]}, {s_name} == {ytrue[2]}")
                
        return False


if __name__ == '__main__':
    postpro = True

    # Real Baseline results with model M3
    if False:
        for mf in [('entropy_add_2', 'add_2'), ('entropy', 'mul')]:
            for exp in ['arity', 'property']:
                print(f"=== {exp} ===")
                print("R1 ['a'], thresholding=False, ta must match")
                ret1 = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=True, acto=False, postprocessing=postpro, mf=mf)

        input("??")

    for mf in [('entropy_add_2', 'add_2'), ('entropy', 'mul')]:
        for exp in ['arity', 'property']:
            print("R1 ['a'], thresholding=False, ta must match")
            ret1 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=False, acto=True, postprocessing=postpro, mf=mf)
            print("R2 ['a'], thresholding=True, ta must match")
            ret2 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=True, postprocessing=postpro, mf=mf)
            print("R3 ['a'], thresholding=False, ta,to,ts must match")
            ret3 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=False, acto=False, postprocessing=postpro, mf=mf)
            print("R4 Only aligned actions: ['a'], thresholding=True, ta,to,ts must match")
            ret4 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=False, postprocessing=postpro, mf=mf)
            print(ret1, ret2, ret3, ret4)

    if True: # Getting only false positives for plot 5
        for mf in [('entropy_add_2', 'add_2')]: #, ('entropy', 'mul')]:
            print(f" === {mf} ===")
            for exp in ['arity', 'property']:
                print(f"--- {exp} ---")
                print("Aligned, thresholding=False")

                fiss = []
                
                f1s = tester_all(exp=[f"{exp}"], alignm=['a'], thr=False, acto=True, postprocessing=postpro, mf=mf)
                fiss.append(f1s)
                print("Aligned, thresholding=True")
                f2s = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=True, postprocessing=postpro, mf=mf)
                fiss.append(f2s)
                
                fiss2 = []
                print("Unaligned, thresholding = False")
                f3s = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=False, acto=True, postprocessing=postpro, mf=mf)
                fiss2.append(f3s)
                print("Unaligned, thresholding = True")
                f4s = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=True, acto=True, postprocessing=postpro, mf=mf)
                fiss2.append(f4s)

                fisss = [fiss, fiss2]

                print("Align x Thres x (BL, ENTROPY, FIXED)")
                print(fisss)
                input("??")
        exit()

    for mf in [('entropy_add_2', 'add_2'), ('entropy', 'mul')]:
        for exp in ['arity', 'property']:
            print(f"=== {exp} ===")
            print("R1 ['a'], thresholding=False, ta must match")
            ret1 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=False, acto=True, postprocessing=postpro, mf=mf)
            # print("R2 ['a'], thresholding=True, ta must match")
            # ret2 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=True, postprocessing=postpro, mf=mf)
            print("R3 ['a'], thresholding=False, ta,to,ts must match")
            ret3 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=False, acto=False, postprocessing=postpro, mf=mf)
            # print("R4 Only aligned actions: ['a'], thresholding=True, ta,to,ts must match")
            # ret4 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=False, postprocessing=postpro, mf=mf)
            print("R5 Checking action match only: ['nag', 'nal', 'nog', 'nol'], thresholding=False, Only ta must match")
            ret5 = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=False, acto=True, postprocessing=postpro, mf=mf)
            
            
            
            # print("R5 Checking action match only: ['nag', 'nal', 'nog', 'nol'], thresholding=True, Only ta must match")
            # ret51 = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=True, acto=True, postprocessing=postpro, mf=mf)
            print("R6 No thresholding (max): ['nag', 'nal', 'nog', 'nol'], thresholding=False, ta,to,ts must match")
            ret6 = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=False, acto=False, postprocessing=postpro, mf=mf)
            # print("R7 All results: ['nag', 'nal', 'nog', 'nol'], thresholding=True, ta,to,ts must match")
            # ret7 = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=True, acto=False, postprocessing=postpro, mf=mf)

            # ret = [ret1,ret2,ret3,ret4,ret5,ret51,ret6,ret7]

            # singlehistplot_customized(np.array(ret), f"real_experiment_{exp}_{mf[0]}__{mf[1]}", xticks=['$ta_{False}^{aligned}$',
            # '$ta_{True}^{aligned}$',
            # '$all_{False}^{aligned}$',
            # '$all_{True}^{aligned}$',
            # '$ta_{False}^{nonaligned}$',
            # '$ta_{True}^{nonaligned}$',
            # '$all_{False}^{nonaligned}$',
            # '$all_{True}^{nonaligned}$'], labels = ['baseline', 'M1', 'M2', 'M3', 'M1 fixed', 'M2 fixed', 'M3 fixed'], plot=True, 
            # title=f'Real Exp. {exp}, Aligned & Non-aligned')

    exit() # Don't evaluate old

    ''' Some old evaluation '''
    postpro = False
    for mf in [('entropy_add_2', 'add_2'), ('entropy', 'mul')]:
        for exp in ['arity', 'property']:
            print(f"=== {exp} ===")
            print("R1 ['a'], thresholding=False, ta must match")
            ret1 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=False, acto=True, postprocessing=postpro, mf=mf)
            print("R2 ['a'], thresholding=True, ta must match")
            ret2 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=True, postprocessing=postpro, mf=mf)
            print("R3 ['a'], thresholding=False, ta,to,ts must match")
            ret3 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=False, acto=False, postprocessing=postpro, mf=mf)
            print("R4 Only aligned actions: ['a'], thresholding=True, ta,to,ts must match")
            ret4 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=False, postprocessing=postpro, mf=mf)
            print("R5 Checking action match only: ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=False, Only ta must match")
            ret5 = tester_all(exp=[f"{exp}"], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=False, acto=True, postprocessing=postpro, mf=mf)
            
            
            
            print("R5 Checking action match only: ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=True, Only ta must match")
            ret51 = tester_all(exp=[f"{exp}"], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=True, acto=True, postprocessing=postpro, mf=mf)
            print("R6 No thresholding (max): ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=False, ta,to,ts must match")
            ret6 = tester_all(exp=[f"{exp}"], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=False, acto=False, postprocessing=postpro, mf=mf)
            print("R7 All results: ['a', 'nag', 'nal', 'nog', 'nol'], thresholding=True, ta,to,ts must match")
            ret7 = tester_all(exp=[f"{exp}"], alignm=['a', 'nag', 'nal', 'nog', 'nol'], thr=True, acto=False, postprocessing=postpro, mf=mf)

            ret = [ret1,ret2,ret3,ret4,ret5,ret51,ret6,ret7]

            singlehistplot_customized(np.array(ret), f"real_experiment_{exp}_{mf[0]}__{mf[1]}", xticks=['$ta_{False}^{aligned}$',
            '$ta_{True}^{aligned}$',
            '$all_{False}^{aligned}$',
            '$all_{True}^{aligned}$',
            '$ta_{False}^{all}$',
            '$ta_{True}^{all}$',
            '$all_{False}^{all}$',
            '$all_{True}^{all}$'], labels = ['baseline', 'M1', 'M2', 'M3', 'M1 fixed', 'M2 fixed', 'M3 fixed'], plot=True, title=f'Real Exp. {exp}')
    
    ''' Some other old evaluation '''
    postpro = True
    for mf in [('entropy', 'mul')]:
        for exp in ['arity', 'property']:
            
            print(f"=== {exp} ===")
            print("R2 ['a'], thresholding=True, ta must match")
            ret2 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=True, postprocessing=postpro, mf=mf)

            print("R4 Only aligned actions: ['a'], thresholding=True, ta,to,ts must match")
            ret4 = tester_all(exp=[f"{exp}"], alignm=['a'], thr=True, acto=False, postprocessing=postpro, mf=mf)


            print("R5 Checking action match only: ['nag', 'nal', 'nog', 'nol'], thresholding=True, Only ta must match")
            ret5 = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=True, acto=True, postprocessing=postpro, mf=mf)
            
            print("R7 All results: ['nag', 'nal', 'nog', 'nol'], thresholding=True, ta,to,ts must match")
            ret7 = tester_all(exp=[f"{exp}"], alignm=['nag', 'nal', 'nog', 'nol'], thr=True, acto=False, postprocessing=postpro, mf=mf)

            ret = [ret2,ret4,ret5,ret7]

            singlehistplot_customized(np.array(ret)[:,0:4], f"real_experiment_{exp}_{mf[0]}__{mf[1]}", xticks=[
            '$ta_{True}^{aligned}$',
            '$all_{True}^{aligned}$',
            '$ta_{True}^{nonaligned}$',
            '$all_{True}^{nonaligned}$'], labels = ['baseline', 'M1', 'M2', 'M3'], plot=True, title=f'Real Exp. {exp}')


