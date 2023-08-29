import globals as g; g.init()
from modality_merger import ProbsVector, SingleTypeModalityMerger, ModalityMerger, UnifiedSentence, MultiProbsVector
from utils import *
import sys, os; sys.path.append("..")
from nlp_new.nlp_utils import make_conjunction, to_default_name
#from nlp_new.sentence_processor_node_new import SentenceProcessor

import numpy as np

def probs_vector_tester():
    ''' ProbsVector holds vector of probabilities and can do additional features 
    '''
    g.template_names = ['pick up', 'place', 'push']
    g.selection_names = ['box', 'big box', 'table']
    g.match_threshold = 0.7
    g.clear_threshold = 0.5
    g.unsure_threshold = 0.2
    g.diffs_threshold = 0.01

    p = [0.9, 0.8, 0.3]
    po = ProbsVector(p)

    assert po.clear_id == [0, 1], f"po.clear_id {po.clear_id}"
    assert po.clear == ['pick up', 'place']
    assert po.unsure == ['push']
    assert po.negative == []
    print(po)

    p=[0.9,0.3,0.2]
    po = ProbsVector(p)
    assert po.clear == ['pick up']
    print(po)
    
    # Match option test
    p = [1.0,0.0,0.0]
    po = ProbsVector(p)
    print(po)
    
    p = [1.0,0.5,0.5]
    po = ProbsVector(p)
    print(po)
    # ask to choose
    p = [0.2,0.2,0.1]
    po = ProbsVector(p)
    print(po)

    p = [0.4,0.3,0.3]
    po = ProbsVector(p)
    print(po)    

def single_modality_tester():
    ''' SingleTypeModalityMerger has two ProbsVectors
    '''
    selection_names = ['red box', 'blue box', 'grey cup']
    cl = [[0.9,0.2,0.1], [0.99,0.1,0.1]]
    cg = [[0.9,0.2,0.1], [0.99,0.1,0.1]]
    print("Object names:\t", selection_names)
    print("Language probs:\t", cl)
    print("Gesture probs:\t", cg)
    selection = SelectionTypeModalityMerger(names=selection_names)
    ret = selection.match(cl, cg)
    print(MultiProbsVector(ret))
    return
    template = SingleTypeModalityMerger()
    cl = [0.9,0.2,0.1]
    cg = [0.9,0.2,0.1]
    ret = template.match(cl, cg)
    print("Action names:\t", g.template_names)
    print("Language probs:\t", cl)
    print("Gesture probs:\t", cg)
    print(ret)

def interactive_plot_tester():
    action = SingleTypeModalityMerger()
    cl = [0.9,0.2,0.1]
    cg = [0.9,0.2,0.1]
    ret = action.match(cl, cg)

    interactive_probs_plotter(cl, cg, g.template_names, g.clear_threshold, g.unsure_threshold, g.diffs_threshold, action)

def modality_merge_tester():
    print(f"{'-' * 5} 1 {'-' * 5}")
    ls = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0],[0.1,0.9,0.0]])
    gs = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0],[0.1,0.9,0.0]])
    mm = ModalityMerger(g.template_names, g.selection_names, ['template', 'selection'])
    print(mm)
    r = mm.feedforward(ls, gs)
    print(r)

    print(f"{'-' * 5} 2 {'-' * 5}")
    ls = UnifiedSentence([0.0,1.0,0.0],[[1.0,0.0,0.0],[0.0,0.0,1.0]])
    gs = UnifiedSentence([0.0,0.8,0.0],[[0.8,0.0,0.0],[0.0,0.0,0.0]])
    
    mm = ModalityMerger(g.template_names, g.selection_names, ['template', 'selection'])
    r = mm.feedforward(ls, gs)
    print(r)

def modality_merge_2_tester():
    ############ Compare types:
    ## PickTask: 'template', 'selection'
    ## PointTask: 'template', 'selection'
    print("Pick Task needs object to detect/ground, when object added, pick task has higher prob")
    print(f"{'-' * 5} 1.1 {'-' * 5}")
    #         ['pick up', 'place', 'push'], ['box', 'big box', 'table']

    g.template_names = ['pick up', 'place', 'push']
    g.selection_names = ['box', 'big box', 'table']

    ls = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0]])
    gs = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0]])
    
    mm = ModalityMerger(g.template_names, g.selection_names, g.compare_types)
    print(mm)
    r = mm.feedforward2(ls, gs)
    print(r)

    print(f"{'-' * 5} 1.2 {'-' * 5}")
    ls = UnifiedSentence([0.9,0.2,0.1],[[0.0,0.0,0.0]])
    gs = UnifiedSentence([0.9,0.2,0.1],[[0.0,0.0,0.0]])
    
    mm = ModalityMerger(g.template_names, g.selection_names, g.compare_types)
    print(mm)
    r = mm.feedforward2(ls, gs)
    print(r)

    print(f"{'-' * 5} 2 {'-' * 5}")
    ls = UnifiedSentence([0.0,1.0,0.0],[[0.0,0.0,0.0]])
    gs = UnifiedSentence([0.0,0.8,0.0],[[0.8,0.0,0.0]])
    
    mm = ModalityMerger(g.template_names, g.selection_names, g.compare_types)
    r = mm.feedforward2(ls, gs)
    print(r)

def test_on_data():
    gestures_data = np.load('/home/imitlearn/crow-base/src/imitrob-hri/imitrob-hri/data/artificial_gestures_01.npy', allow_pickle=True)
    speech_data = np.load('/home/imitlearn/crow-base/src/imitrob-hri/imitrob-hri/data/artificial_speech_01.npy', allow_pickle=True)
    results_data = np.load('/home/imitlearn/crow-base/src/imitrob-hri/imitrob-hri/data/artificial_results_01.npy', allow_pickle=True)
    
    acc = 0
    for gs, ls, trueres in zip(gestures_data, speech_data, results_data):
        print("gta", gs.target_template, " lta ", ls.target_template, " gts: ", gs.target_selections, " lts: ", ls.target_selections)

        mm = ModalityMerger(g.template_names, g.selection_names, g.compare_types)
        r = mm.feedforward2(ls, gs)
        print(r)
        print(r.activated, trueres[0])
        print(r.activated == trueres[0])
        if r.activated == trueres[0]:
            acc +=1

    print(f"Final acc: {acc/40 *100}%")
    
    print(results_data)


def names_to_default():
    language_template_name = utils.ct_name_to_default_name(language_template_name, ct='template') # "pick that" -> "pick up"
    language_templates = [utils.ct_name_to_default_name(name, ct='template') for name in self.get_language_templates()] # all templates available



def test_on_data2():
    dataset = np.load(os.path.expanduser('~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_02.npy'), allow_pickle=True)
    
    ''' Set configuration '''
    g.compare_types = ['template', 'selections']
    g.match_threshold = 0.48
    g.clear_threshold = 0.44
    g.unsure_threshold = 0.2
    g.diffs_threshold = 0.01

    g.object_properties = {
        'box': {
            'reachable': True,
            'pickable': True,
        },
        'big box': {
            'reachable': True,
            'pickable': False,
        },
        'table': {
            'reachable': True ,
            'pickable': False,
        },


        'Cube': {
            'reachable': True,
            'pickable': True,
        },
        'Peg': {
            'reachable': True,
            'pickable': True,
        },
        'aruco box': {
            'reachable': True,
            'pickable': True,
        },

        'Cup': {
            'reachable': True,
            'pickable': True,
        },

    }

    g.task_property_penalization = {
        'PickTask': {
            'reachable': 0.8,
            'pickable': 0.0,
        }, 'PointTask': {
            'reachable': 1.0,
            'pickable': 1.0,
        },
    }

    acc = 0
    nsamples = len(dataset)
    for n,sample in enumerate(dataset):
        print(f"{'*' * 10} {n}th sample {'*' * 10}")
        ls = sample['xl'] 
        gs = sample['xg']
        y_template = sample['y_template']
        y_selection = sample['y_selection']

        g.template_names, t_g, t_l = make_conjunction(gs.target_template_names, ls.target_template_names, \
                            gs.target_template, ls.target_template, ct='template')

        for template in ['pick', 'point', 'PutTask']:
            if to_default_name(template) not in g.template_names:
                g.template_names = np.append(g.template_names, to_default_name(template))
                t_g = np.append(t_g, 0.0)
                t_l = np.append(t_l, 0.0)

        g.selection_names, o_g, o_l = make_conjunction(gs.target_selection_names, ls.target_selection_names, \
                            gs.target_selections, ls.target_selections, ct='selection')
        gs_extended = UnifiedSentence(t_g, target_selections=o_g, target_template_names=g.template_names, target_selection_names=g.selection_names)
        ls_extended = UnifiedSentence(t_l, target_selections=o_l, target_template_names=g.template_names, target_selection_names=g.selection_names)
        
        
        mm = ModalityMerger(g.template_names, g.selection_names, g.compare_types)
        t, s = mm.feedforward2(ls_extended, gs_extended)
        
        if t.activated == y_template: # and t.activated == y_selection:
            acc +=1
        else:
            print("ls", ls)
            print("gs", gs)
            print("y_template", y_template)
            print("y_selection", y_selection)

            print(t)

    print(f"Final acc: {acc/nsamples*100}%")


if __name__ == '__main__':
    #print("1. Single probs vector tester: \n")
    #probs_vector_tester()

    #print("2. Selection type tester: \n")
    #single_modality_tester()

    #print("3. Modality merge tester: \n")
    #modality_merge_tester()

    #print("4. Interactive plot tester: \n")
    #interactive_plot_tester()

    #print("5. Modality merge 2 tester: \n")
    #modality_merge_2_tester()


    #test_on_data()
    test_on_data2()
