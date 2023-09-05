
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
        # Sample True option
        y_template = np.random.choice(c.templates)

        if 'selections' in create_template(y_template).compare_types:
            y_selection = np.random.choice(c.selections)
        else:
            y_selection = None
        if 'storages' in create_template(y_template).compare_types:
            y_storages = np.random.choice(c.storages)
        else:
            y_storages = None

        

        scene = get_random_scene(y_selection, gen_params)

        G = {
            'template': ProbsVector(*get_random_names_and_likelihoods(names=c.templates, true_name=y_template, gen_params=gen_params, min_ch=1), c),
            'selections': ProbsVector(*get_random_names_and_likelihoods(names=c.selections, true_name=y_selection, gen_params=gen_params, min_ch=0), c),
            'storages': ProbsVector(*get_random_names_and_likelihoods(names=c.storages, true_name=y_storages, gen_params=gen_params, min_ch=0), c),
        }
        L = {
            'template': ProbsVector(*get_random_names_and_likelihoods(names=c.templates, true_name=y_template, gen_params=gen_params, min_ch=1), c),
            'selections':ProbsVector(*get_random_names_and_likelihoods(names=c.selections, true_name=y_selection, gen_params=gen_params, min_ch=0), c),
            'storages':ProbsVector(*get_random_names_and_likelihoods(names=c.storages, true_name=y_storages, gen_params=gen_params, min_ch=0), c),
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
            'only_single_object_action_performable': False
        })
    elif dataset_n == 2:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration1(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'only_single_object_action_performable': False
        })
    elif dataset_n == 3:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.01,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.01,
            'complementary': False,
            'only_single_object_action_performable': False
        })
    elif dataset_n == 4:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'only_single_object_action_performable': False
        })
    elif dataset_n == 5:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration3(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.01,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.01,
            'complementary': False,
            'only_single_object_action_performable': False
        })
    elif dataset_n == 6:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration3(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'only_single_object_action_performable': False
        })
    elif dataset_n == 7:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.01,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.01,
            'complementary': True,
            'only_single_object_action_performable': False
        })
    elif dataset_n == 8:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': True,
            'only_single_object_action_performable': False
        })
    elif dataset_n == 9:
        dataset = generate_dataset(gen_params = {
            'configuration': Configuration2(),
            'unsure_prob_mu': 0.5,
            'unsure_prob_sigma': 0.1,
            'activated_prob_mu': 0.9,
            'activated_prob_sigma': 0.1,
            'complementary': False,
            'only_single_object_action_performable': True
        })
    np.save(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_0{dataset_n}.npy'), dataset)
