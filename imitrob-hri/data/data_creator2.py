
import os
import sys;sys.path.append("..")
from merging_modalities.modality_merger import UnifiedSentence
import numpy as np

template_list = ['pick', 'put', 'point']
selection_names_list = ['potted meat can', 'tomato soup can', 'bowl', 'Cube', 'Peg', 'box', 'big box', 'aruco box']

def normal(mu, sigma):
    ''' normal cropped (0 to 1) '''
    return np.clip(np.random.normal(mu, sigma), 0, 1)
def add_if_not_there(a, b):
    if b not in a:
        return np.append(a, b)
    return a


class Configuration():
    def __init__(self):
        #self.template_names = ['pick up', 'place', 'push']
        self.template_names = ['PickTask', 'PointTask', 'PutTask']
        self.selection_names = ['box', 'big box', 'table']
        self.compare_types = ['template', 'selections']

        self.match_threshold = 0.4
        self.clear_threshold = 0.34
        self.unsure_threshold = 0.15
        self.diffs_threshold = 0.01

        self.object_properties = {
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

        self.task_property_penalization = {
            'PickTask': {
                'reachable': 0.8,
                'pickable': 0.0,
            }, 'PointTask': {
                'reachable': 1.0,
                'pickable': 1.0,
            },
        }
        self.DEBUG = False


unsure_prob_mu = 0.5
unsure_prob_sigma = 0.1

activated_prob_mu = 0.9
activated_prob_sigma = 0.05

dataset = []
for i_sample in range(100):
    # Sample True option
    y_template = np.random.choice(template_list)
    y_selection = np.random.choice(selection_names_list)

    ## Templates
    # Get observed gesture templates
    ct_template_gesture_chosen_names = np.random.choice(template_list, size=np.random.randint(1, len(template_list) ), replace=False)
    ct_template_gesture_chosen_names = add_if_not_there(ct_template_gesture_chosen_names, y_template)
    # Give all likelihoods unsure probs.
    ct_template_gesture_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_template_gesture_chosen_names))]
    # Give the true value the activated prob
    ct_template_gesture_likelihood[np.where(ct_template_gesture_chosen_names == y_template)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    
    ct_template_language_chosen_names = np.random.choice(template_list, size=np.random.randint(1, len(template_list) ), replace=False)
    ct_template_language_chosen_names = add_if_not_there(ct_template_language_chosen_names, y_template)
    ct_template_language_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_template_language_chosen_names))]
    ct_template_language_likelihood[np.where(ct_template_language_chosen_names == y_template)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)

    ## Objects (Selection)
    ct_object_gesture_chosen_names = np.random.choice(selection_names_list, size=np.random.randint(1, len(selection_names_list) ), replace=False)
    ct_object_gesture_chosen_names = add_if_not_there(ct_object_gesture_chosen_names, y_selection)
    ct_object_gesture_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_object_gesture_chosen_names))]
    ct_object_gesture_likelihood[np.where(ct_object_gesture_chosen_names == y_selection)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    
    ct_object_language_chosen_names = np.random.choice(selection_names_list, size=np.random.randint(1, len(selection_names_list) ), replace=False)
    ct_object_language_chosen_names = add_if_not_there(ct_object_language_chosen_names, y_selection)
    ct_object_language_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_object_language_chosen_names))]
    ct_object_language_likelihood[np.where(ct_object_language_chosen_names == y_selection)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    
    x_g = UnifiedSentence(ct_template_gesture_likelihood, ct_object_gesture_likelihood, \
        target_template_names=ct_template_gesture_chosen_names, target_selection_names=ct_object_gesture_chosen_names)
    x_l = UnifiedSentence(ct_template_language_likelihood, ct_object_language_likelihood, \
        target_template_names=ct_template_language_chosen_names, target_selection_names=ct_object_language_chosen_names)

    sample = {
        'xl': x_l, 
        'xg': x_g,
        'y_template': y_template,
        'y_selection': y_selection,
    }
    if i_sample == 0:
        sample['config'] = Configuration()
    dataset.append(sample)



np.save(os.path.expanduser('~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_02.npy'), dataset)

def get_sample():
    dataset = np.load(os.path.expanduser('~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_02.npy'), allow_pickle=True)
    return dataset[0]
