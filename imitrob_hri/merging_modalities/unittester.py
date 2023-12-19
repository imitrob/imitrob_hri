import unittest

import sys; sys.path.append("..")
from imitrob_nlp.nlp_utils import *


import numpy as np

class TestPlayingWithWords(unittest.TestCase):

    def test_to_default_name(self):
        for i in range(1000):
            random_key_name = np.random.choice(list(template_name_synonyms.keys()))
            random_key_id = np.random.randint(0, len(template_name_synonyms[random_key_name]))
            random_name = template_name_synonyms[random_key_name][random_key_id]
            
            self.assertEqual(template_name_to_id(random_name), random_key_name, "Wrong")    
            self.assertEqual(to_default_name(random_name), template_name_synonyms[random_key_name][0], "Wrong")

    def test_make_conjunction(self):
    
        gesture_templates = ['pick', 'put']
        language_templates = ['point', 'fetch']
        gesture_likelihoods = [0.9, 0.4]
        language_likelihoods = [0.5, 0.2]
        
        unique_list, lp_unified, gp_unified = make_conjunction(gesture_templates, language_templates, gesture_likelihoods, language_likelihoods)

        self.assertEqual(set(unique_list),set(['pick', 'put','point', 'fetch']))
        self.assertEqual(set(gp_unified), set([0.9, 0.4, 0.0, 0.0]))
        self.assertEqual(set(lp_unified), set([0.0, 0.0, 0.5, 0.2]))


if __name__ == '__main__':
    unittest.main()




