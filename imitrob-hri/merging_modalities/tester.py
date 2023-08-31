from configuration import Configuration
from modality_merger import ProbsVector, SingleTypeModalityMerger, ModalityMerger, MultiProbsVector
from utils import *
import sys, os; sys.path.append("..")
from nlp_new.nlp_utils import make_conjunction, to_default_name
#from nlp_new.sentence_processor_node_new import SentenceProcessor

import numpy as np

def probs_vector_tester():
    ''' ProbsVector holds vector of probabilities and can do additional features 
    '''
    c = Configuration()

    c.template_names = ['pick up', 'place', 'push']
    c.selection_names = ['box', 'big box', 'table']
    c.match_threshold = 0.7
    c.clear_threshold = 0.5
    c.unsure_threshold = 0.2
    c.diffs_threshold = 0.01

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
    print("Action names:\t", c.template_names)
    print("Language probs:\t", cl)
    print("Gesture probs:\t", cg)
    print(ret)

def interactive_plot_tester():
    action = SingleTypeModalityMerger()
    cl = [0.9,0.2,0.1]
    cg = [0.9,0.2,0.1]
    ret = action.match(cl, cg)

    interactive_probs_plotter(cl, cg, c.template_names, c.clear_threshold, c.unsure_threshold, c.diffs_threshold, action)

def modality_merge_tester():
    print(f"{'-' * 5} 1 {'-' * 5}")
    c = Configuration()
    ls = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0],[0.1,0.9,0.0]])
    gs = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0],[0.1,0.9,0.0]])
    mm = ModalityMerger(c.template_names, c.selection_names, ['template', 'selection'])
    print(mm)
    r = mm.feedforward(ls, gs)
    print(r)

    print(f"{'-' * 5} 2 {'-' * 5}")
    ls = UnifiedSentence([0.0,1.0,0.0],[[1.0,0.0,0.0],[0.0,0.0,1.0]])
    gs = UnifiedSentence([0.0,0.8,0.0],[[0.8,0.0,0.0],[0.0,0.0,0.0]])
    
    mm = ModalityMerger(c.template_names, c.selection_names, ['template', 'selection'])
    r = mm.feedforward(ls, gs)
    print(r)

def modality_merge_2_tester():
    ############ Compare types:
    ## PickTask: 'template', 'selection'
    ## PointTask: 'template', 'selection'
    print("Pick Task needs object to detect/ground, when object added, pick task has higher prob")
    print(f"{'-' * 5} 1.1 {'-' * 5}")
    #         ['pick up', 'place', 'push'], ['box', 'big box', 'table']

    c.template_names = ['pick up', 'place', 'push']
    c.selection_names = ['box', 'big box', 'table']

    ls = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0]])
    gs = UnifiedSentence([0.9,0.2,0.1],[[0.9,0.1,0.0]])
    
    mm = ModalityMerger(c.template_names, c.selection_names, c.compare_types)
    print(mm)
    r = mm.feedforward2(ls, gs)
    print(r)

    print(f"{'-' * 5} 1.2 {'-' * 5}")
    ls = UnifiedSentence([0.9,0.2,0.1],[[0.0,0.0,0.0]])
    gs = UnifiedSentence([0.9,0.2,0.1],[[0.0,0.0,0.0]])
    
    mm = ModalityMerger(c.template_names, c.selection_names, c.compare_types)
    print(mm)
    r = mm.feedforward2(ls, gs)
    print(r)

    print(f"{'-' * 5} 2 {'-' * 5}")
    ls = UnifiedSentence([0.0,1.0,0.0],[[0.0,0.0,0.0]])
    gs = UnifiedSentence([0.0,0.8,0.0],[[0.8,0.0,0.0]])
    
    mm = ModalityMerger(c.template_names, c.selection_names, c.compare_types)
    r = mm.feedforward2(ls, gs)
    print(r)

if __name__ == '__main__':
    print("1. Single probs vector tester: \n")
    probs_vector_tester()

    print("2. Selection type tester: \n")
    single_modality_tester()

    print("3. Modality merge tester: \n")
    modality_merge_tester()

    print("4. Interactive plot tester: \n")
    interactive_plot_tester()

    print("5. Modality merge 2 tester: \n")
    modality_merge_2_tester()
