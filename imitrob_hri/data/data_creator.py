
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
        for i_sample in range(c.samples):
            y_template, y_selection, y_storages = None, None, None
            # Scene depends on feasibility
            # The scene should be able to be created as feasible for an action, but there might be scene on which nothing can be done which are discarded
            iteration = 0
            while y_template is None:
                iteration += 1
                if iteration > 100000: raise Exception(f"Didn't found any configuration! debug: {scene} {get_random_feasible_triplet(scene)} {get_templates_decisive_based_on_properties(names=c.templates, true_name=y_template, min_ch=1, scene=scene)}")

                scene = get_random_scene(c)
                y_template, y_selection, y_storages = get_random_feasible_triplet(scene)

                # check that any decidable
                if rp == 'fake_properties_decidable_wrt_true' and len(c.templates)>3:
                    tdp = get_templates_decisive_based_on_properties(names=c.templates, true_name=y_template, min_ch=1, scene=scene)
                    if len(tdp) == 0:
                        #print("discarded sample")
                        y_template = None
                
            det_fun = gen_params['det_fun']
            noise_fun = gen_params['noise_fun']
            
            if rp == 'one_bigger':
                rp_M_C = (('one_bigger', '-', '-'), ('-', '-', '-'))
            else:
                rp_M_C = ((rp, '-', '-'), (rp, '-', '-'))
            
            G = {
                'template': ProbsVector(*generate_probs(names=c.templates,           true_name=y_template, det_fun=det_fun, min_ch=1, sim_table=c.sim_table_gesture, scene=scene, regulation_policy=rp_M_C[0][0], noise_fun=noise_fun), c),
                'selections': ProbsVector(*generate_probs(names=scene.selection_names, true_name=y_selection, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_gesture_objects, scene=scene, regulation_policy=rp_M_C[0][1], noise_fun=noise_fun), c),
                'storages': ProbsVector(*generate_probs(names=scene.storage_names,   true_name=y_storages, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_gesture_storages, scene=scene, regulation_policy=rp_M_C[0][2], noise_fun=noise_fun), c),
            }
            L = {
                'template': ProbsVector(*generate_probs(names=c.templates,           true_name=y_template, det_fun=det_fun, min_ch=1, sim_table=c.sim_table_language, scene=scene, regulation_policy=rp_M_C[1][0], noise_fun=noise_fun), c),
                'selections':ProbsVector(*generate_probs(names=scene.selection_names, true_name=y_selection, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_language_objects, scene=scene, regulation_policy=rp_M_C[1][1], noise_fun=noise_fun), c),
                'storages':ProbsVector(*generate_probs(names=scene.storage_names,   true_name=y_storages, det_fun=det_fun, min_ch=0, sim_table=c.sim_table_language_storages, scene=scene, regulation_policy=rp_M_C[1][2], noise_fun=noise_fun), c),
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


def gen_dataset(c,n,d):
    config = {'c1': Configuration1(), 'c2':Configuration2(), 'c3': Configuration3()}[c]
    noise = {'n1': (nm.NormalModel(0.9, 0.01), nm.NormalModel(0.0,0.05)), 
             'n2': (nm.gesture_det_model, nm.gesture_noise_model),
             'n3': (nm.gesture_det_model, nm.gesture_noise_model2),
             'n4': (nm.gesture_det_model, nm.gesture_noise_model3),}[n]
    policies_str = d[1:]
    policies_list = [
        '-',
        'fake_arity_decidable_wrt_true',
        'fake_properties_decidable_wrt_true',
        'undecidable_wrt_true',
        'one_bigger',
    ]
    policies = []
    for char in policies_str:
        policies.append(policies_list[int(char)-1])

    dataset = generate_dataset(gen_params = {
        'configuration': config,
        'det_fun': noise[0],
        'noise_fun': noise[1],
        'complementary': False,
        'policies': policies,
    })

    np.save(f'{os.path.dirname(os.path.abspath(__file__))}/saves/artificial_dataset_{c}_{n}_{d}.npy', dataset)

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    if dataset_name == 'all':
        for c in ['c1', 'c2', 'c3']:
            for n in ['n1', 'n2', 'n3', 'n4']:
                for d in ['D1', 'D2', 'D3', 'D4', 'D5']:
                    gen_dataset(c,n,d)
    
    else:
        c,n,d = dataset_name.split("_")
        gen_dataset(c,n,d)
