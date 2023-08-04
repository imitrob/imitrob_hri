
import globals as g; g.init()
from modality_merger import ProbsVector, SingleTypeModalityMerger, SelectionTypeModalityMerger, ModalityMerger, UnifiedSentence, MultiProbsVector
from utils import *

import numpy as np

def probs_vector_tester():
    ''' ProbsVector holds vector of probabilities and can do additional features 
    '''
    action_names = ['pick up', 'place', 'push']
    match_threshold = 0.8
    clear_threshold = 0.7
    unsure_threshold = 0.2

    p = [0.9, 0.8, 0.2]
    po = ProbsVector(p)

    assert po.clear_id == [0, 1]
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
    object_names = ['red box', 'blue box', 'grey cup']
    cl = [[0.9,0.2,0.1], [0.99,0.1,0.1]]
    cg = [[0.9,0.2,0.1], [0.99,0.1,0.1]]
    aor = [2]
    print("Object names:\t", object_names)
    print("Language probs:\t", cl)
    print("Gesture probs:\t", cg)
    selection = SelectionTypeModalityMerger(names=object_names)
    ret = selection.match(cl, cg, aor)
    print(MultiProbsVector(ret))
    return
    action = SingleTypeModalityMerger()
    cl = [0.9,0.2,0.1]
    cg = [0.9,0.2,0.1]
    ret = action.match(cl, cg)
    print("Action names:\t", g.action_names)
    print("Language probs:\t", cl)
    print("Gesture probs:\t", cg)
    print(ret)

def interactive_plot_tester():
    action = SingleTypeModalityMerger()
    cl = [0.9,0.2,0.1]
    cg = [0.9,0.2,0.1]
    ret = action.match(cl, cg)

    interactive_probs_plotter(cl, cg, g.action_names, g.clear_threshold, g.unsure_threshold, g.diffs_threshold, action)

def modality_merge_tester():
    print(f"{'-' * 5} 1 {'-' * 5}")
    ls = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0],[0.1,0.9,0.0]])
    gs = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0],[0.1,0.9,0.0]])
    action_names = g.action_names
    object_names = g.object_names
    mm = ModalityMerger(action_names, object_names)
    print(mm)
    r = mm.feedforward(ls, gs)
    print(r)

    print(f"{'-' * 5} 2 {'-' * 5}")
    ls = UnifiedSentence([0.0,1.0,0.0],[[1.0,0.0,0.0],[0.0,0.0,1.0]])
    gs = UnifiedSentence([0.0,0.8,0.0],[[0.8,0.0,0.0],[0.0,0.0,0.0]])
    action_names = g.action_names
    object_names = g.object_names
    mm = ModalityMerger(action_names, object_names)
    r = mm.feedforward(ls, gs)
    print(r)

def modality_merge_2_tester():
    ############ Compare types:
    ## PickTask: 'action', 'selection'
    ## PointTask: 'action', 'selection'
    print("Pick Task needs object to detect/ground, when object added, pick task has higher prob")
    print(f"{'-' * 5} 1.1 {'-' * 5}")
    #         ['pick up', 'place', 'push'], ['box', 'big box', 'table']

    ls = UnifiedSentence([0.9,0.2,0.1],[0.9,0.1,0.0])
    gs = UnifiedSentence([0.9,0.2,0.1],[0.9,0.1,0.0])
    action_names = g.action_names
    object_names = g.object_names
    mm = ModalityMerger(action_names, object_names, g.compare_types)
    print(mm)
    r = mm.feedforward2(ls, gs)
    print(r)

    print(f"{'-' * 5} 1.2 {'-' * 5}")
    ls = UnifiedSentence([0.9,0.2,0.1],[0.0,0.0,0.0])
    gs = UnifiedSentence([0.9,0.2,0.1],[0.0,0.0,0.0])
    action_names = g.action_names
    object_names = g.object_names
    mm = ModalityMerger(action_names, object_names, g.compare_types)
    print(mm)
    r = mm.feedforward2(ls, gs)
    print(r)

    print(f"{'-' * 5} 2 {'-' * 5}")
    ls = UnifiedSentence([0.0,1.0,0.0],[0.0,0.0,0.0])
    gs = UnifiedSentence([0.0,0.8,0.0],[0.8,0.0,0.0])
    action_names = g.action_names
    object_names = g.object_names
    mm = ModalityMerger(action_names, object_names, g.compare_types)
    r = mm.feedforward2(ls, gs)
    print(r)

def test_on_data():
    gestures_data = np.load('/home/imitlearn/crow-base/src/imitrob-hri/imitrob-hri/data/artificial_gestures_01.npy', allow_pickle=True)
    speech_data = np.load('/home/imitlearn/crow-base/src/imitrob-hri/imitrob-hri/data/artificial_speech_01.npy', allow_pickle=True)
    for gs, ls in zip(gestures_data, speech_data):
        action_names = g.action_names
        object_names = g.object_names
        mm = ModalityMerger(action_names, object_names, g.compare_types)
        r = mm.feedforward2(ls, gs)
        return 

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
    modality_merge_2_tester()
