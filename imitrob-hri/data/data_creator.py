
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from merging_modalities.modality_merger import MMSentence, ProbsVector
import numpy as np
from data.datagen_utils import *
from nlp_new.nlp_utils import create_template
from merging_modalities.configuration import *
import merging_modalities.noise_model as nm

def generate_dataset(gen_params):

    c = gen_params['configuration']
    dataset = []
    for rp in gen_params['policies']:
        for i_sample in range(10000):
            y_template, y_selection, y_storages = None, None, None
            # Scene depends on feasibility
            iteration = 0
            while y_template is None:
                iteration += 1
                if iteration > 10000: raise Exception("Iterations")

                scene = get_random_scene(c)
                y_template, y_selection, y_storages = get_random_feasible_triplet(scene)
                
            det_fun = gen_params['det_fun']
            noise_fun = gen_params['noise_fun']
            
            G = {
                'template': ProbsVector(*generate_probs(names=c.templates,           true_name=y_template, det_fun=det_fun, min_ch=1, sim_table=c.sim_table_gesture, scene=scene, regulation_policy=rp, noise_fun=noise_fun), c),
                'selections': ProbsVector(*generate_probs(names=scene.selection_names, true_name=y_selection, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_gesture_objects, scene=scene, regulation_policy='-', noise_fun=noise_fun), c),
                'storages': ProbsVector(*generate_probs(names=scene.storage_names,   true_name=y_storages, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_gesture_storages, scene=scene, regulation_policy='-', noise_fun=noise_fun), c),
            }
            L = {
                'template': ProbsVector(*generate_probs(names=c.templates,           true_name=y_template, det_fun=det_fun, min_ch=1, sim_table=c.sim_table_language, scene=scene, regulation_policy=rp, noise_fun=noise_fun), c),
                'selections':ProbsVector(*generate_probs(names=scene.selection_names, true_name=y_selection, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_language_objects, scene=scene, regulation_policy='-', noise_fun=noise_fun), c),
                'storages':ProbsVector(*generate_probs(names=scene.storage_names,   true_name=y_storages, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_language_storages, scene=scene, regulation_policy='-', noise_fun=noise_fun), c),
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


def gen_dataset(c,n,p):
    config = {'c1': Configuration11(), 'c2':Configuration12(), 'c3': Configuration13()}[c]
    noise = {'n1': (nm.NormalModel(0.9, 0.01), nm.NormalModel(0.0,0.05)), 
             'n2': (nm.gesture_det_model, nm.gesture_noise_model),
             'n3': (nm.gesture_det_model, nm.gesture_noise_model2),}[n]
    policies_str = p[1:]
    policies_list = [
        '-',
        'fake_arity_decidable_wrt_true',
        'undecidable_wrt_true',
        'fake_properties_decidable_wrt_true',
    ]
    policies = []
    for char in policies_str:
        policies.append(policies_list[int(char)])

    dataset = generate_dataset(gen_params = {
        'configuration': config,
        'det_fun': noise[0],
        'noise_fun': noise[1],
        'complementary': False,
        'policies': policies,
    })

    np.save(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_{c}_{n}_{p}.npy'), dataset)

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    if dataset_name == 'all':
        for c in ['c1', 'c2', 'c3']:
            for n in ['n1', 'n2', 'n3']:
                for p in ['p0','p1','p2','p3']:
                    gen_dataset(c,n,p)
    
    else:
        c,n,p = dataset_name.split("_")
        gen_dataset(c,n,p)

if False: # Old series archive
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
