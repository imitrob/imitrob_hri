
# import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from imitrob_hri.merging_modalities.modality_merger import MMSentence, ProbsVector
import numpy as np
from imitrob_hri.data.datagen_utils2 import *
from imitrob_hri.data.scene3_def import *
from imitrob_hri.imitrob_nlp.TemplateFactory import create_template
from merging_modalities.configuration import *
import merging_modalities.noise_model as nm
import time

def req_prop_to_observation(prop, sign):
    assert sign in ['-','+'], f"sign should be + or -, it is {sign}"
    assert isinstance(prop, str), f"property in string form, it is type: {type(prop)}"
    
    RATE_DIFF = 0.1
    
    if prop == 'pickable':
        threshold_being_unpickable = 0.12 # [m] <- robot's gripper max opened distance 
        
        if sign == '+': # property is needed to be pickable
            value = threshold_being_unpickable * (1-RATE_DIFF)
        elif sign == '-':
            value = threshold_being_unpickable * (1+RATE_DIFF)
        return 'size', value

    if prop == 'reachable':
        threshold_radius_being_unreachable = 0.6 # [m]
        
        if sign == '+': # property is needed to be pickable
            value = [threshold_radius_being_unreachable * (1-RATE_DIFF)] * 3
        elif sign == '-':
            value = [threshold_radius_being_unreachable * (1+RATE_DIFF)] * 3
        return 'position', value
    
    if prop == 'stackable':
        ''' Depending on the top surface of the object '''
        threshold_roundtop_being_unstackable = 0.1 # [belief rate]
        
        if sign == '+': # property is needed to be pickable
            value = threshold_roundtop_being_unstackable * (1-RATE_DIFF)
        elif sign == '-':
            value = threshold_roundtop_being_unstackable * (1+RATE_DIFF)
        return 'roundness-top', value

    if prop == 'pushable':
        threshold_weight_being_unpushable = 2 # [kg]
        
        if sign == '+': # property is needed to be pickable
            value = threshold_weight_being_unpushable * (1-RATE_DIFF)
        elif sign == '-':
            value = threshold_weight_being_unpushable * (1+RATE_DIFF)
        return 'weight', value

    if prop == 'full-stack':
        
        if sign == '+': # property is needed to be full
            value = 1
        elif sign == '-':
            value = 0
        return 'contain_item', value

    if prop == 'full-liquid':
        threshold_capacity_being_full = 0.7 # [norm. rate being full]

        if sign == '+': # property is needed to be pickable
            value = threshold_capacity_being_full * (1-RATE_DIFF)
        elif sign == '-':
            value = threshold_capacity_being_full * (1+RATE_DIFF)
        return 'contains', value

    if prop == 'glued':
        
        if sign == '+': # property is needed to be glued
            value = 1
        elif sign == '-':
            value = 0
        return 'glued', value    
    
    raise Exception(f"Not right property name: {prop}")

def get_minimal_valid_scene_for_triplet(c, triplet, other_prop_oposite):
    y_template, y_selection, y_storage = triplet
    template_o = create_template(y_template)

    o = None
    if y_selection is not None:
        # find a single object with valid properties
        observations = {'name': y_selection, 'types': ['liquid container', 'object']}
        # Generate observations for object, if observation is needed for the object to have property for the template, it is fulfilled
        # If the observation is not needed for the object, we doesn't care (random), or
        # other_prop_oposite=True: we care, and assign observation as oposite
        for prop in c.properties:
            
            if prop in template_o.feasibility_requirements['+']:
                obs_name, obs_value = req_prop_to_observation(prop, '+')
                observations[obs_name] = obs_value
            elif other_prop_oposite:
                obs_name, obs_value = req_prop_to_observation(prop, '-')
                observations[obs_name] = obs_value
            
            if prop in template_o.feasibility_requirements['-']:
                obs_name, obs_value = req_prop_to_observation(prop, '-')
                observations[obs_name] = obs_value
            elif other_prop_oposite:
                obs_name, obs_value = req_prop_to_observation(prop, '+')
                observations[obs_name] = obs_value
            
        o = Object3(observations)
        
    s = None
    if y_storage is not None:
        # find a single storage with valid properties
        observations = {'name': y_storage, 'types': ['liquid container', 'object']}
        for prop in c.properties:
            if prop in template_o.feasibility_requirements['+']:
                obs_name, obs_value = req_prop_to_observation(prop, '+')
                observations[obs_name] = obs_value
            elif other_prop_oposite:
                obs_name, obs_value = req_prop_to_observation(prop, '-')
                observations[obs_name] = obs_value
            
            if prop in template_o.feasibility_requirements['-']:
                obs_name, obs_value = req_prop_to_observation(prop, '-')
                observations[obs_name] = obs_value
            elif other_prop_oposite:
                obs_name, obs_value = req_prop_to_observation(prop, '+')
                observations[obs_name] = obs_value
        
        o = Object3(observations)
        
    return o, s

def get_observations_cannot_be_done_by_this_action(c, target_action):
    template_o = create_template(target_action)
    
    # get requirements (properties) for the action
    positive_requirements = deepcopy(template_o.feasibility_requirements['+'])
    negative_requirements = deepcopy(template_o.feasibility_requirements['-'])
    
    # invert the requirements, get object that CANNOT be done by this action
    # this is not right
    # c.properties - positive_requirements
    positive_requirements_ = [x for x in c.properties if not x in positive_requirements or positive_requirements.remove(x)]
    # c.properties - negative_requirements
    negative_requirements_ = [x for x in c.properties if not x in negative_requirements or negative_requirements.remove(x)]
    
    # convert them to the observations
    observations = {}
    for prop in positive_requirements_:
        obs_name, obs_value = req_prop_to_observation(prop, '+')
        observations[obs_name] = obs_value
    for prop in negative_requirements_:
        obs_name, obs_value = req_prop_to_observation(prop, '-')
        observations[obs_name] = obs_value
    
    return observations
    
def gen_random_object(name):
    observations = {
        'name': name, 
        'size': np.random.random() * 0.5, # [m]
        'position': [np.random.random(), np.random.random(), 0.0], # [m,m,m]
        'roundness-top': np.random.random() * 0.2, # [normalized belief rate]
        'weight': np.random.random() * 4, # [kg]
        'contains': np.random.random(), # normalized rate being full 
        'contain_item': np.random.randint(0,2), # normalized rate being full 
        'types': ['liquid container', 'object'],
        'glued': np.random.randint(0,2),
    }
    return Object3(observations)

def generate_dataset2(gen_params):

    c = gen_params['configuration']
    dataset = []
    for dataset_policy in gen_params['dataset_policies']:
        for i_sample in range(c.samples):
            # 1: true random triplet (y_template, y_selection, y_storages) (str, str, str)
            triplet = get_random_triplet(c)
            y_template, y_selection, y_storages = triplet
            
            O, S = [], []
            o, s = get_minimal_valid_scene_for_triplet(c, triplet, other_prop_oposite=True)
            if o is not None: O.append(o)
            if s is not None: S.append(s)
            
            # 2: true scene
            remaining_n_objects = get_n_objects(c) - len(O)
            remaining_n_storages = get_n_storages(c) - len(S)

            # Generate other objects
            if o is not None: # target_action depends on the object
                remaining_object_names = np.random.choice([x for x in c.objects if not x in [y_selection] or [y_selection].remove(x)], size=remaining_n_objects, replace=False)
                for name in remaining_object_names: 
                    # Objects that cannot be done by this action
                    obsv = get_observations_cannot_be_done_by_this_action(c, triplet[0])
                    obsv['name'] = name
                    obsv['types'] = ['liquid container', 'object']
                    o = Object3(obsv)
                    O.append(o)
            else:
                remaining_object_names = np.random.choice(c.objects, size=remaining_n_objects, replace=False)
                # generate random objects
                for name in remaining_object_names: 
                    O.append(gen_random_object(name))
            
            if s is not None: # target_action depends on the object
                remaining_storage_names = np.random.choice([x for x in c.storages if not x in [y_storages] or [y_storages].remove(x)], size=remaining_n_storages, replace=False)
                for name in remaining_storage_names:
                    # Storages that cannot be done by this action
                    obsv = get_observations_cannot_be_done_by_this_action(c, triplet[0])
                    obsv['name'] = name
                    obsv['types'] = ['liquid container', 'object']
                    s = Object3(obsv)
                    S.append(s)
            else:
                remaining_storage_names = np.random.choice(c.storages, size=remaining_n_storages, replace=False)
                # generate random objects
                for name in remaining_storage_names: 
                    S.append(gen_random_object(name))
            
                
            scene = Scene3(O, S, template_names=c.templates)
            
            
            # 2.1: check that any decidable (previous fails it tried until finds feasible),
            #      only for D3
            if dataset_policy == 'fake_properties_decidable_wrt_true' and len(c.templates)>3:
                tdp = get_templates_decisive_based_on_properties(names=c.templates, true_name=y_template, min_ch=1, scene=scene)
                if len(tdp) == 0:
                    #print("discarded sample")
                    y_template = None
            
            det_fun = gen_params['det_fun']
            noise_fun = gen_params['noise_fun']
            
            # I needed to add this workaround argument for  for D5
            if dataset_policy == 'one_bigger': # D5
                rp_M_C = ((dataset_policy, '-', '-'), ('-', '-', '-'))
            else:
                rp_M_C = ((dataset_policy, '-', '-'), (dataset_policy, '-', '-'))
            
            # 3. Generate probs
            G = {
                'template': ProbsVector(*generate_probs(
                    names=c.templates,
                    true_name=y_template,
                    det_fun=det_fun,
                    min_ch=1,
                    sim_table=c.sim_table_gesture,
                    scene=scene,
                    regulation_policy=rp_M_C[0][0],
                    noise_fun=noise_fun), c),
                'selections': ProbsVector(*generate_probs(
                    names=c.objects, #scene.selection_names,
                    true_name=y_selection,
                    det_fun=det_fun,
                    min_ch=0,
                    sim_table=c.sim_table_gesture_objects,
                    scene=scene,
                    regulation_policy=rp_M_C[0][1],
                    noise_fun=noise_fun), c),
                'storages': ProbsVector(*generate_probs(
                    names=c.storages, #scene.storage_names,
                    true_name=y_storages,
                    det_fun=det_fun,
                    min_ch=0,
                    sim_table=c.sim_table_gesture_storages,
                    scene=scene,
                    regulation_policy=rp_M_C[0][2],
                    noise_fun=noise_fun), c),
            }
            L = {
                'template': ProbsVector(*generate_probs(
                    names=c.templates,
                    true_name=y_template,
                    det_fun=det_fun,
                    min_ch=1,
                    sim_table=c.sim_table_language,
                    scene=scene,
                    regulation_policy=rp_M_C[1][0],
                    noise_fun=noise_fun), c),
                'selections':ProbsVector(*generate_probs(
                    names=c.objects, #scene.selection_names,
                    true_name=y_selection,
                    det_fun=det_fun,
                    min_ch=0,
                    sim_table=c.sim_table_language_objects,
                    scene=scene,
                    regulation_policy=rp_M_C[1][1],
                    noise_fun=noise_fun), c),
                'storages':ProbsVector(*generate_probs(
                    names=c.storages, #scene.storage_names,
                    true_name=y_storages,
                    det_fun=det_fun,
                    min_ch=0,
                    sim_table=c.sim_table_language_storages,
                    scene=scene,
                    regulation_policy=rp_M_C[1][2],
                    noise_fun=noise_fun), c),
            }

            # I don't use this for now
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


def gen_dataset2(c,n,d):
    config = {'c1': Configuration2_1(), 'c2':Configuration2_2(), 'c3': Configuration2_3()}[c]
    noise = {'n1': (nm.NormalModel(1.0, 0.0), nm.NormalModel(0.0,0.0)), 
             'n2': (nm.gesture_det_model, nm.gesture_noise_model2),
             'n3': (nm.gesture_det_model, nm.gesture_noise_model4),}[n]
    policies_str = d[1:]
    policies_list = [
        '-',
        'fake_arity_decidable_wrt_true',
        'fake_properties_decidable_wrt_true',
        'one_bigger',
    ]
    policies = []
    for char in policies_str:
        policies.append(policies_list[int(char)-1])

    dataset = generate_dataset2(gen_params = {
        'configuration': config,
        'det_fun': noise[0],
        'noise_fun': noise[1],
        'complementary': False,
        'dataset_policies': policies,
    })

    np.save(f'{os.path.dirname(os.path.abspath(__file__))}/saves/artificial_dataset2_{c}_{n}_{d}.npy', dataset)

if __name__ == '__main__':
    
    dataset_name = sys.argv[1]
    if dataset_name == 'all':
        for c in ['c1', 'c2', 'c3']:
            for n in ['n1', 'n2', 'n3']:
                for d in ['D1', 'D2', 'D3', 'D4']:
                    gen_dataset2(c,n,d)
    
    else:
        c,n,d = dataset_name.split("_")
        gen_dataset2(c,n,d)
