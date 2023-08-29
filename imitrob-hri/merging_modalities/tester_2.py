import globals as g; g.init()
from modality_merger import ProbsVector, SingleTypeModalityMerger, SelectionTypeModalityMerger, ModalityMerger, UnifiedSentence, MultiProbsVector, penalize_properties, get_ct_properties
from utils import *
import sys; sys.path.append("..")

import numpy as np


def test_penalize_properties():
    g.DEBUG = True

    print(f"{'*' * 10} 1 {'*' * 10}")
    g.selection_properties['box']['reachable'] = False
    template = 'PickTask' 
    property_name = 'reachable'
    compare_type = 'selection'
    S_naive_c = ProbsVector([1.0,0.0,0.0], ['box', 'big box', 'Cup'])
    S_naive = {
        'template': ProbsVector([1.0,0.0,0.0], ['PickTask', 'PointTask', 'PushTask']),
        'selection': ProbsVector([1.0,0.0,0.0], ['box', 'big box', 'Cup'])
    }

    pnlz = penalize_properties(template, property_name, compare_type, S_naive_c)

    #print(pnlz)

    print(f"{'*' * 10} 2 {'*' * 10}")
    compare_types = ['template', 'selection']

    beta = 1.0
    for nct, compare_type in enumerate(compare_types): # selections, storages, distances, ...
        
        for property_name in get_ct_properties(compare_type):
            print("penalize_properties", template, property_name, compare_type, S_naive[compare_type])
            # check properties, penalize non-compatible ones
            b = penalize_properties(template, property_name, compare_type, S_naive[compare_type])
            beta *= b

    print(beta)




if __name__ == '__main__':
    test_penalize_properties()
