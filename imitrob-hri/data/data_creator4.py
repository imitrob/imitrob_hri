
import os
import sys;sys.path.append("..")
from merging_modalities.modality_merger import MMSentence, ProbsVector
import numpy as np

template_list = ['pick', 'put', 'point']
selection_names_list = ['potted meat can', 'tomato soup can', 'bowl', 'Cube', 'Peg', 'box', 'big box', 'aruco box']
storage_names_list = ['green box', 'abstract zone']

def normal(mu, sigma):
    ''' normal cropped (0 to 1) '''
    return np.clip(np.random.normal(mu, sigma), 0, 1)
def add_if_not_there(a, b):
    if b not in a:
        return np.append(a, b)
    return a


class Configuration():
    def __init__(self):
        self.ct_names = {
            'template': ['PickTask', 'PointTask', 'PutTask'],
            'selections': ['box', 'big box', 'table'],
            'storages': ['green box', 'abstract zone']
        }

        self.match_threshold = 0.55
        self.clear_threshold = 0.5
        self.unsure_threshold = 0.25
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

        self.get_ct_properties = {
            'selection': ['reachable', 'pickable'],
            '': [],
        }

        self.epsilon = 0.9
        self.gamma = 0.5

        self.DEBUG = False


unsure_prob_mu = 0.5
unsure_prob_sigma = 0.1

activated_prob_mu = 0.9
activated_prob_sigma = 0.05

c = Configuration()
dataset = []
for i_sample in range(100):
    # Sample True option
    y_template = np.random.choice(template_list)
    y_selection = np.random.choice(selection_names_list)
    y_storage = np.random.choice(storage_names_list)

    language_enabled = bool(np.random.randint(0,2))
    gesture_enabled = not language_enabled

    ## Templates
    # Get observed gesture templates
    if gesture_enabled:
        ct_template_gesture_chosen_names = np.random.choice(template_list, size=np.random.randint(1, len(template_list) ), replace=False)
        ct_template_gesture_chosen_names = add_if_not_there(ct_template_gesture_chosen_names, y_template)
        # Give all likelihoods unsure probs.
        ct_template_gesture_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_template_gesture_chosen_names))]
        # Give the true value the activated prob
        ct_template_gesture_likelihood[np.where(ct_template_gesture_chosen_names == y_template)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    else:
        ct_template_gesture_chosen_names = np.array([])
        ct_template_gesture_likelihood = np.array([])
    if language_enabled:
        ct_template_language_chosen_names = np.random.choice(template_list, size=np.random.randint(1, len(template_list) ), replace=False)
        ct_template_language_chosen_names = add_if_not_there(ct_template_language_chosen_names, y_template)
        ct_template_language_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_template_language_chosen_names))]
        ct_template_language_likelihood[np.where(ct_template_language_chosen_names == y_template)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    else:
        ct_template_language_chosen_names = np.array([])
        ct_template_language_likelihood = np.array([])

    ## Objects (Selection)
    if gesture_enabled:
        ct_object_gesture_chosen_names = np.random.choice(selection_names_list, size=np.random.randint(1, len(selection_names_list) ), replace=False)
        ct_object_gesture_chosen_names = add_if_not_there(ct_object_gesture_chosen_names, y_selection)
        ct_object_gesture_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_object_gesture_chosen_names))]
        ct_object_gesture_likelihood[np.where(ct_object_gesture_chosen_names == y_selection)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    else:
        ct_object_gesture_chosen_names = np.array([])
        ct_object_gesture_likelihood = np.array([])
    
    if language_enabled:
        ct_object_language_chosen_names = np.random.choice(selection_names_list, size=np.random.randint(1, len(selection_names_list) ), replace=False)
        ct_object_language_chosen_names = add_if_not_there(ct_object_language_chosen_names, y_selection)
        ct_object_language_likelihood = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(ct_object_language_chosen_names))]
        ct_object_language_likelihood[np.where(ct_object_language_chosen_names == y_selection)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    else:
        ct_object_language_chosen_names = np.array([])
        ct_object_language_likelihood = np.array([])

    ## Storages
    if gesture_enabled:
        storage_g_names = np.random.choice(storage_names_list, size=np.random.randint(1, len(storage_names_list) ), replace=False)
        storage_g_names = add_if_not_there(storage_g_names, y_storage)
        storage_g_p = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(storage_g_names))]
        storage_g_p[np.where(storage_g_names == y_storage)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    else:
        storage_g_names = np.array([])
        storage_g_p = np.array([])

    if language_enabled:
        storage_l_names = np.random.choice(storage_names_list, size=np.random.randint(1, len(storage_names_list) ), replace=False)
        storage_l_names = add_if_not_there(storage_l_names, y_storage)
        storage_l_p = [normal(unsure_prob_mu, unsure_prob_sigma) for i in range(len(storage_l_names))]
        storage_l_p[np.where(storage_l_names == y_storage)[0][0]] = normal(activated_prob_mu, activated_prob_sigma)
    else:
        storage_l_names = np.array([])
        storage_l_p = np.array([])

    G = {
        'template': ProbsVector(ct_template_gesture_likelihood, ct_template_gesture_chosen_names, c),
        'selections':ProbsVector(ct_object_gesture_likelihood, ct_object_gesture_chosen_names, c),
        'storages':ProbsVector(storage_g_p, storage_g_names, c),
    }
    L = {
        'template': ProbsVector(ct_template_language_likelihood, ct_template_language_chosen_names, c),
        'selections':ProbsVector(ct_object_language_likelihood, ct_object_language_chosen_names, c),
        'storages':ProbsVector(storage_l_p, storage_l_names, c),
    }
    s = MMSentence(L, G)

    sample = {
        'x_sentence': s,
        'y': {
            'template': y_template,
            'selections': y_selection,
            'storages': y_storage,
        }
    }
    if i_sample == 0:
        sample['config'] = c
    dataset.append(sample)



np.save(os.path.expanduser('~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_04.npy'), dataset)

def get_sample():
    dataset = np.load(os.path.expanduser('~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_04.npy'), allow_pickle=True)
    return dataset[0]
