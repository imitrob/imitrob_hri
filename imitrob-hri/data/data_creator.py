
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from merging_modalities.modality_merger import MMSentence, ProbsVector
import numpy as np
from data.datagen_utils import *
from nlp_new.nlp_utils import create_template
from merging_modalities.configuration import *

def generate_dataset(gen_params):

    c = gen_params['configuration']
    dataset = []
    for i_sample in range(10000):
        scene = get_random_scene(c)

        y_template, y_selection, y_storages = get_random_feasible_triplet(scene)
        '''
        # Sample True option
        y_template = np.random.choice(c.templates)
        if 'selections' in create_template(y_template).compare_types and c.selections != []:
            y_selection = np.random.choice(c.selections)
        else:
            y_selection = None
        if 'storages' in create_template(y_template).compare_types and c.storages != []:
            y_storages = np.random.choice(c.storages)
        else:
            y_storages = None
        '''
        assert isinstance(y_template, (str, type(None))), f"wrong type {y_template}"
        assert isinstance(y_selection, (str, type(None))), f"wrong type {y_selection}"
        assert isinstance(y_storages, (str, type(None))), f"wrong type {y_storages}"

        if gen_params['penalize_scene_objects']:
            scene = penalize_scene_objects(scene, y_selection, y_storages)
        
        G = {
            'template': ProbsVector(*get_random_names_and_likelihoods(names=c.templates, true_name=y_template, gen_params=gen_params, min_ch=1), c),
            'selections': ProbsVector(*get_random_names_and_likelihoods(names=scene.selection_names, true_name=y_selection, gen_params=gen_params, min_ch=0), c),
            'storages': ProbsVector(*get_random_names_and_likelihoods(names=scene.storage_names, true_name=y_storages, gen_params=gen_params, min_ch=0), c),
        }
        L = {
            'template': ProbsVector(*get_random_names_and_likelihoods(names=c.templates, true_name=y_template, gen_params=gen_params, min_ch=1), c),
            'selections':ProbsVector(*get_random_names_and_likelihoods(names=scene.selection_names, true_name=y_selection, gen_params=gen_params, min_ch=0), c),
            'storages':ProbsVector(*get_random_names_and_likelihoods(names=scene.storage_names, true_name=y_storages, gen_params=gen_params, min_ch=0), c),
        }

        if gen_params['complementary']:
            for k in G.keys():
                if bool(np.random.randint(0,2)):
                    L[k] = ProbsVector(np.array([]), np.array([]), c)
                else:
                    G[k] = ProbsVector(np.array([]), np.array([]), c)
                
        s = MMSentence(L, G)

        sample = {
            'x_scene': scene,
            'x_sentence': s,
            'config': c,
            'y': {
                'template': y_template,
                'selections': y_selection,
                'storages': y_storages,
            }
        }
        dataset.append(sample)
    return dataset

def penalize_scene_objects(scene, y_selection, y_storages):


    ######### selection #######
    for o in scene.selections:
        if o.name != y_selection:
            ### Assign unsatisfyable observations
            o.observations['size'] += 1 # [m]
            o.observations['position'] += [1.0, 1.0, 1.0] # [m,m,m]
            o.observations['roundness-top'] += 0.3 # [normalized belief rate]
            o.observations['weight'] += 5 # [kg]
            o.observations['contains'] = 1 # normalized rate being full 
            o.observations['glued'] = True

    for o in scene.storages:
        if o.name != y_storages:
            ### Assign unsatisfyable observations
            o.observations['size'] += 1 # [m]
            o.observations['position'] += [1.0, 1.0, 1.0] # [m,m,m]
            o.observations['roundness-top'] += 0.3 # [normalized belief rate]
            o.observations['weight'] += 5 # [kg]
            o.observations['contains'] = 1 # normalized rate being full 
            o.observations['glued'] = True

    return scene 


if __name__ == '__main__':
    dataset_n = int(sys.argv[1])
    if dataset_n == 1:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration1(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.01,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.01,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 2:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration1(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 3:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.01,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.01,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 4:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 5:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration3(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.01,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.01,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 6:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration3(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 7:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.01,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.01,
            'complementary': True,
            'penalize_scene_objects': False
        })
    elif dataset_n == 8:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': True,
            'penalize_scene_objects': False
        })
    elif dataset_n == 9:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'penalize_scene_objects': True
        })
    elif dataset_n == 10:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.2,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.2,
            'complementary': False,
            'penalize_scene_objects': True
        })
    elif dataset_n == 11: # higher noise
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration1(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.2,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.2,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 12: # higher noise
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.2,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.2,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 13: # higher noise
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration3(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.2,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.2,
            'complementary': False,
            'penalize_scene_objects': False
        })
    elif dataset_n == 14: # higher noise
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.2,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.2,
            'complementary': True,
            'penalize_scene_objects': False
        })
    elif dataset_n == 15: # higher noise
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.2,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.2,
            'complementary': False,
            'penalize_scene_objects': True
        })
    np.save(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_0{dataset_n}.npy'), dataset)
