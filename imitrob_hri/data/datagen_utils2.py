import numpy as np
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from merging_modalities.utils import cc
from imitrob_hri.imitrob_nlp.TemplateFactory import create_template
from copy import deepcopy
from merging_modalities import noise_model
from imitrob_hri.data.scene3_def import add_if_not_there

from merging_modalities.utils import fdo

def get_random_triplet(c):
    ''' approach 2
        
    Parameters:
        c (Configuration)
        
    Returns:
        None if no triplet found
        (template, selection, storage) (Str, Str, Str): True meants triplet by the user
    '''
    object = None
    storage = None

    template = np.random.choice(c.templates)
    template_o = create_template(template)
    requirements = deepcopy(template_o.mm_pars_compulsary) # requirements to fill
    if 'template' in requirements:
        requirements.remove('template')
    else:
        raise Exception("template not in requirements for template: {template.name}")
    if 'selections' in requirements:
        object = np.random.choice(c.objects)
        if object is None: raise Exception("This cannot happen, config is wrongly designed")
    if 'storages' in requirements:
        storage = np.random.choice(c.storages)
        if storage is None: raise Exception("This cannot happen, config is wrongly designed")

    return template, object, storage


def generate_probs(names, true_name, det_fun, min_ch, sim_table, scene, regulation_policy, noise_fun):
    ''' approach v2
        - Note: chosen_names_subset is full list
        - We have names of category items, true detected name
    Parameters:
        names (String[]): e.g. template names
        true_name (String): Y, e.g. 'pick'
        det_fun (function): det_fun() generates probability based model
        min_ch (Int): minimum chosen itmes (there is min. 1 action, min. 0 objects)
        sim_table (Float[][]): Similarity table
        scene (Scene3())
        regulation_policy (String)
    '''
    if len(names) == 0 or true_name is None: 
        return np.array([]), np.array([])

    # 1. Get observed gesture templates: [pick, pour] chosen from all e.g. [pick, pour, point, push, ...]
    # chosen_names_subset =     np.random.choice(names, size=np.random.randint(min_ch, len(names) ), replace=False)
    chosen_names_subset = np.array(names)
    # 2. If true_name == 'stack' -> is not in chosen_names_subset -> add it 
    chosen_names_subset = add_if_not_there(chosen_names_subset, true_name)
    # 3. Give all likelihoods ones [1.0, 1.0, 1.0], [pick, pour, stack]
    P = [1.0] * len(chosen_names_subset)
    # 4. Give the true value the activated prob
    id_activated = np.where(chosen_names_subset == true_name)[0][0]
    activated_normal = det_fun()
    # 5.1. Prepare sim_table_subset
    name_ids = []
    for name in chosen_names_subset: # 
        name_ids.append(names.index(name))

    sim_table_subset = np.array(sim_table)[np.ix_(name_ids, name_ids)]

    # 5. fill rest of likelihoods based on similarity table
    #       [pick, pour, stack]
    # pick  [1   , 0,8 , 0.7  ]
    # pour  [0.8 , 1   , 0.6  ]
    # stack [0.7 , 0.6 , 1    ] -> [0.7*~0.9, 0.6*~0.9, 1.0*~0.9] 
    l = len(sim_table_subset[id_activated])

    # 6. Model added noise
    added_noise = noise_fun(l) #normal(np.zeros(l),noise_fun(l) * np.ones(l))
    added_noise_regulation_policy = noise_fun() #normal(0, noise_fun())

    #print("added_noise", added_noise)
    #print("activated_normal", activated_normal)
    #print("sim_table_subset[id_activated] * activated_normal", sim_table_subset[id_activated] * activated_normal)
    P = np.clip(sim_table_subset[id_activated] * activated_normal + added_noise, 0 , 1)
    #print("P", P)
    #input()

    # D2
    if regulation_policy == 'fake_arity_decidable_wrt_true': # All templates which can be recognized has activate type
        chosen_names_subset_ = get_templates_decisive_based_on_arity(names, true_name, min_ch, scene)
        
        for chosen_name_subset_ in chosen_names_subset_:
            # the ones which are not decisive based on arity
            if chosen_name_subset_ not in chosen_names_subset:
                raise Exception("This won't happen, when there is not subsets")
                chosen_names_subset = np.append(chosen_names_subset,chosen_name_subset_)
                P = np.append(P, activated_normal)
            else:
                # It is decisive a 
                id = list(chosen_names_subset).index(chosen_name_subset_)
                P[id] = np.clip(activated_normal + added_noise[id], 0,1)

    # D5
    elif regulation_policy == 'undecidable_wrt_true':
        # 1. choose random subset 
        
        chosen_names_subset_1 = get_templates_decisive_based_on_arity(names, true_name, min_ch, scene)
        chosen_names_subset_2 = get_templates_decisive_based_on_properties(names, true_name, min_ch, scene)
        chosen_names_subset_ = list(set(chosen_names_subset_1 + chosen_names_subset_2))

        chosen_names_subset_inv = [nm for nm in names if nm not in chosen_names_subset_]

        for chosen_name_subset_ in chosen_names_subset_inv:
            if chosen_name_subset_ not in chosen_names_subset:
                raise Exception("This won't happen, when there is not subsets")
                chosen_names_subset = np.append(chosen_names_subset,chosen_name_subset_)
                P = np.append(P, activated_normal)
            else:
                id = list(chosen_names_subset).index(chosen_name_subset_)
                P[id] = np.clip(activated_normal + added_noise[id],0,1)

    # D3
    if regulation_policy == 'fake_properties_decidable_wrt_true':
        chosen_names_subset_ = get_templates_decisive_based_on_properties(names, true_name, 
        min_ch, scene)

        #chosen_names_subset_1 = get_templates_decisive_based_on_arity(names, true_name, min_ch, scene)

        # print("regulation_policy", regulation_policy)
        # print("chosen_names_subset_", chosen_names_subset_)
        
        for chosen_name_subset_ in chosen_names_subset_:
            if chosen_name_subset_ not in chosen_names_subset: 
                # This shouldn't happen        
                raise Exception("This should not happen")     
                chosen_names_subset = np.append(chosen_names_subset,chosen_name_subset_)
                P = np.append(P, activated_normal)
            else:
                id = list(chosen_names_subset).index(chosen_name_subset_)
                P[id] = np.clip(activated_normal + added_noise_regulation_policy,0,1)

    # D4
    if regulation_policy == 'one_bigger':
        
        chosen_names_subset_1 = get_templates_decisive_based_on_arity(names, true_name, min_ch, scene)
        chosen_names_subset_2 = get_templates_decisive_based_on_properties(names, true_name, min_ch, scene)
        chosen_names_subset_ = list(set(chosen_names_subset_1 + chosen_names_subset_2))

        chosen_names_subset_inv = [nm for nm in names if nm not in chosen_names_subset_]
        # choose one undecisive
        if len(chosen_names_subset_inv) > 0:
            chosen_name_subset_ = chosen_names_subset_inv[0]
        else:
            # Should return None
            return None, None
            chosen_name_subset_ = chosen_names_subset_[0]

        if chosen_name_subset_ not in chosen_names_subset:
            raise Exception("This won't happen, when there is not subsets")
            chosen_names_subset = np.append(chosen_names_subset,chosen_name_subset_)
            P = np.append(P, 1.0)
        else:
            id = list(chosen_names_subset).index(chosen_name_subset_)
            P *= 0.9 # one needs to be bigger
            P[id] = 1.0

    return P, chosen_names_subset



def generate_probs_old(names, true_name, gen_params, min_ch):
    ''' approach v1 
    '''
    activated_mu = gen_params['activated_prob_mu']
    activated_sigma = gen_params['activated_prob_sigma']
    unsure_mu = gen_params['unsure_prob_mu']
    unsure_sigma = gen_params['unsure_prob_sigma']

    if len(names) == 0 or true_name is None: 
        return np.array([]), np.array([])
    # Get observed gesture templates
    chosen_names_subset = np.random.choice(names, size=np.random.randint(min_ch, len(names) ), replace=False)
    chosen_names_subset = add_if_not_there(chosen_names_subset, true_name)
    # Give all likelihoods unsure probs.
    P = [normal(unsure_mu, unsure_sigma) for _ in range(len(chosen_names_subset))]
    # Give the true value the activated prob
    P[np.where(chosen_names_subset == true_name)[0][0]] = normal(activated_mu, activated_sigma)

    

    return P, chosen_names_subset


def get_templates_decisive_based_on_arity(names, true_name, min_ch, scene):
    # 1. set of item include true_name + other templates which we can distinguish
    t = create_template(true_name)
    # Get template compulsory parameters
    t_cts = {}
    for tt in scene.templates:
        t_cts[tt.name] = tt.mm_pars_compulsary
    
    chosen_names_subset = []
    for k in t_cts.keys():
        if t_cts[k] != t.mm_pars_compulsary:
            chosen_names_subset.append(k)    
    
    if true_name not in chosen_names_subset: chosen_names_subset.append(true_name)
    # 2. assign probs
    #act_p = normal(gen_params['activated_prob_mu'], gen_params['activated_prob_sigma'])
    #P = act_p * np.ones(len(chosen_names_subset))
    # fdo.arity['template'].append() 
    return chosen_names_subset

def get_templates_decisive_based_on_properties(names, true_name, min_ch, scene):
    ''' Returns distinguishable templates based on properties
    Note: May return only true_name template only
    Parameters:
    Return: template names (String[])
    '''

    templates = scene.templates
    unfeasible_templates = []
    for template in templates:
        
        feasible = is_action_is_feasible_given_this_scene(template, scene)
        
        if not feasible and template.complexity > 0:
            unfeasible_templates.append(template.name)


    return unfeasible_templates

def is_action_is_feasible_given_this_scene(template, scene):
    feasible = False
    # if any combination feasible -> It is False
    if len(scene.selections) == 0 and len(scene.storages) == 0:
        
        feasible = template.is_feasible(o=None, s=None)
    elif len(scene.selections) > 0 and len(scene.storages) == 0:

        for o in scene.selections:
            if template.is_feasible(o, s=None):
                feasible = True

    elif len(scene.selections) == 0 and len(scene.storages) > 0:
        raise Exception("resolve here, we have storage but no object!")
    
    elif len(scene.selections) > 0 and len(scene.storages) > 0:
        
        for o in scene.selections:
            for s in scene.storages:
                if template.is_feasible(o, s):
                    feasible = True
    
    else: 
        raise Exception("This is strange!")
    
    return feasible