import numpy as np
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from merging_modalities.utils import cc
from nlp_new.nlp_utils import create_template
from copy import deepcopy

def normal(mu, sigma):
    ''' normal cropped (0 to 1) '''
    return np.clip(np.random.normal(mu, sigma), 0, 1)


def add_if_not_there(a, b):
    if b not in a:
        return np.append(a, b)
    return a


class Object3():
    def __init__(self, observations = {
            'name': 'box',
            'size': 0.1, # [m]
            'position': [0.0, 0.0, 0.0], # [m,m,m]
            'roundness-top': 0.1, # [normalized belief rate]
            'weight': 1, # [kg]
            'contains': 0.1, # normalized rate being full 
            'contain_item': False, # how many items contains
            'types': ['liquid container', 'object'],
            'glued': False
        }):
        self.name = observations['name']

        self.observations = observations

        self.properties = {}
        self.properties['glued'] = self.glued
        if 'object' in observations['types']:
            self.properties['pickable'] = self.pickable
            self.properties['reachable'] = self.reachable
            self.properties['stackable'] = self.stackable
            self.properties['pushable'] = self.pushable
        if 'liquid container' in observations['types']:
            self.properties['full-stack'] = self.full_stack
            self.properties['full-liquid'] = self.full_liquid

    def is_type(self, typ):
        return typ in self.observations['types']

    def pickable(self, r=4):
        threshold_being_unpickable = 0.12 # [m] <- robot's gripper max opened distance 

        penalization = self.sigmoid(self.observations['size'], center=threshold_being_unpickable)
        eval = self.observations['size'] <= threshold_being_unpickable
        return round(penalization,r), eval

    def reachable(self, r=4):
        threshold_radius_being_unreachable = 0.6 # [m]
        penalization, eval = 1.0, True
        for x in self.observations['position']:
            p = self.sigmoid(x, center=threshold_radius_being_unreachable)
            penalization *= p
            e = (x <= threshold_radius_being_unreachable)
            eval &= e
        return round(penalization,r), eval
    
    def stackable(self, r=4):
        ''' Depending on the top surface of the object '''
        threshold_roundtop_being_unstackable = 0.1 # [belief rate]
        
        penalization = self.sigmoid(self.observations['roundness-top'], center=threshold_roundtop_being_unstackable)
        eval = (self.observations['roundness-top'] <= threshold_roundtop_being_unstackable)
        return round(penalization,r), eval

    def pushable(self, r=4):
        threshold_weight_being_unpushable = 2 # [kg]
        
        penalization = self.sigmoid(self.observations['weight'], center=threshold_weight_being_unpushable)
        eval = (self.observations['weight'] <= threshold_weight_being_unpushable)
        return round(penalization,r), eval

    def full_stack(self, r=4):
        return round(int(self.observations['contain_item']),r), int(self.observations['contain_item'])

    def full_liquid(self, r=4):
        threshold_capacity_being_full = 0.7 # [norm. rate being full]
        
        penalization = self.sigmoid(self.observations['contains'], center=threshold_capacity_being_full)
        eval = (self.observations['contains'] <= threshold_capacity_being_full)
        return round(penalization,r), eval     

    def glued(self, r=4):
        return round(int(self.observations['glued']),r), int(self.observations['glued'])
    
    ''' Modelling functions: '''
    @staticmethod
    def gaussian(x, sigma=0.2):
        return np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))

    @staticmethod
    def sigmoid(x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + np.exp((center-x)*(-tau)))
    
    def __str__(self):
        return f"{cc.F}{self.name}{cc.E}: s: {round(self.observations['size'],2)}, p: {np.round(self.observations['position'],2)}, roundness: {round(self.observations['roundness-top'],2)}, weight: {round(self.observations['weight'],2)}, contains: {round(self.observations['contains'],2)}, glued: {self.observations['glued']}\nProperties: {cc.H}Glued:{cc.E} {self.glued(2)}, {cc.H}Pickable:{cc.E} {self.pickable(2)}, {cc.H}Reachable:{cc.E} {self.reachable(2)}, {cc.H}Stackable {cc.E}{self.stackable(2)}, {cc.H}Pushable:{cc.E} {self.pushable(2)}, {cc.H}Full stack:{cc.E} {self.full_stack(2)} full liquid: {self.full_liquid(2)}\n"
    
    
class Scene3():
    def __init__(self, selections, storages, template_names=['move-up', 'release', 'stop', 'pick', 'point', 'push', 'unglue', 'pour', 'put-into', 'stack']):
        self.templates = []
        for template_name in template_names:
            self.templates.append( create_template(template_name) )

        self.selections = selections
        self.storages = storages

    def get_template(self, name):
        for t in self.templates:
            if t.name == name:
                return t
        raise Exception(f"Template {name} not found!")

    def get_object(self, name):
        for n, o in enumerate(self.selections):
            if o.name == name:
                return o
        raise Exception(f"Object with name {name} not found! ({self.selections})")

    @property
    def selection_names(self):
        return [o.name for o in self.selections]

    @property
    def storage_names(self):
        return [o.name for o in self.storages]
    
    def __str__(self):
        s = ''
        for sel in self.selections:
            s += f"{sel}"
        
        s2 = ''
        for sto in self.storages:
            s2 += f"{sto}"

        return f"{cc.H}Selections{cc.E}:\n{s}{cc.H}Storages{cc.E}:\n{s2}"


def get_random_scene(c, object_name_list=['potted meat can', 'tomato soup can', 'bowl', 'box', 'big box', 'paper', 'wrench', 'glued wrench'], storage_name_list=['paper box', 'abstract marked zone', 'out of table', 'on the table']):

    # Based on configuration, choose how many objects and storages to generate in the scene
    if c.scene_gen_config['selections_n'][0] == 'uniform':
        low, high = c.scene_gen_config['selections_n'][1:3]
        nobjs = np.random.randint(low, high)
    else: raise Exception("TODO ADD OTHER THAN UNIFORM PROB DISTR.")
        
    if c.scene_gen_config['storages_n'][0] == 'uniform':
        low, high = c.scene_gen_config['storages_n'][1:3]
        nstgs = np.random.randint(low, high)
    else: raise Exception("TODO ADD OTHER THAN UNIFORM PROB DISTR.")

    objects = []
    storages = []

    object_chosen_names = list(np.random.choice(object_name_list, size=nobjs, replace=False))
    
    for object_name in object_chosen_names:
        assert isinstance(object_name, str), f"object name is not string {object_name}"
        observations = {
            'name': object_name, 
            'size': np.random.random() * 0.5, # [m]
            'position': [np.random.random(), np.random.random(), 0.0], # [m,m,m]
            'roundness-top': np.random.random() * 0.2, # [normalized belief rate]
            'weight': np.random.random() * 4, # [kg]
            'contains': np.random.random(), # normalized rate being full 
            'contain_item': np.random.randint(0,2), # normalized rate being full 
            'types': ['liquid container', 'object'],
            'glued': np.random.randint(0,2),
        }
        objects.append(Object3(observations))

    storage_chosen_names = list(np.random.choice(storage_name_list, size=nstgs, replace=False))
    
    for storage_name in storage_chosen_names:
        assert isinstance(storage_name, str), f"object name is not string {storage_name}"
        observations = {
            'name': storage_name, 
            'size': np.random.random() * 0.5, # [m]
            'position': [np.random.random(), np.random.random(), 0.0], # [m,m,m]
            'roundness-top': np.random.random() * 0.2, # [normalized belief rate]
            'weight': np.random.random() * 4, # [kg]
            'contains': np.random.random(), # normalized rate being full 
            'contain_item': np.random.randint(0,2), # normalized rate being full 
            'types': ['liquid container', 'object'],
            'glued': np.random.randint(0,2),
        }
        storages.append(Object3(observations))

    scene = Scene3(objects, storages, template_names=c.templates)

    return scene

def get_random_feasible_triplet(scene, template_request=None):
    ''' approach 1
    Returns:
        None if no triplet found
    '''
    object = None
    storage = None

    requirements = ['do']
    if template_request is None:
        
        template = None
        arange = list(range(len(scene.templates)))
        np.random.shuffle(arange)
        for t_n in arange:
            t = scene.templates[t_n]
            #if t.is_feasible(): # possibility to restrict template based on scene config
            template = t
            break
        
        if template is None:
            raise Exception()
    
    else:
        assert isinstance(template_request, str)
        template = scene.get_template(template_request)
    requirements = deepcopy(template.compare_types) # requirements to fill
    
    if 'template' in requirements:
        requirements.remove('template')

    def choose_selection(object, storage):
        arange_selections = list(range(len(scene.selections)))
        np.random.shuffle(arange_selections)

        for o_n in arange_selections:
            o = scene.selections[o_n]
            if template.is_feasible(o):
                object = o
                requirements.remove('selections')
                break
        return object

    def choose_storage(object, storage):
        arange_storages = list(range(len(scene.storages)))
        np.random.shuffle(arange_storages)

        for s_n in arange_storages:
            s = scene.storages[s_n]
            if template.is_feasible(object, s):
                storage = s
                requirements.remove('storages')
                break
        return storage

    if 'selections' in requirements:
        object = choose_selection(object, storage)
        if object is None: return None, None, None
    if 'storages' in requirements:
        storage = choose_storage(object, storage)
        if storage is None: return None, None, None

    return triplet_to_name(template, object, storage)

def triplet_to_name(template, object, storage):
    if template is not None:
        template = template.name
    if object is not None:
        object = object.name
    if storage is not None:
        storage = storage.name
    return template, object, storage



def generate_probs(names, true_name, activated_mu, activated_sigma, min_ch, sim_table, scene, regulation_policy, noise_sigma):
    ''' approach v2
    Parameters:
        names (String[]): e.g. template names
        true_name (String): Y
        gen_params (Dict): { ... parameters ... }
        min_ch (Int): minimum chosen itmes
        sim_table (Float[][]): Similarity table
        names_table 
    '''
    if len(names) == 0 or true_name is None: 
        return np.array([]), np.array([])
    elif regulation_policy == 'not aligned':
        pass # do something

    # 1. Get observed gesture templates: [pick, pour] chosen from all [pick, pour, point, push, ...]
    chosen_names_subset = np.random.choice(names, size=np.random.randint(min_ch, len(names) ), replace=False)
    # 2. If true_name == 'stack' -> is not in chosen_names_subset -> add it 
    chosen_names_subset = add_if_not_there(chosen_names_subset, true_name)
    # 3. Give all likelihoods zeros [1.0, 1.0, 1.0], [pick, pour, stack]
    P = [1.0] * len(chosen_names_subset)
    # 4. Give the true value the activated prob -> [1.0, 1.0, 1.0], [pick, pour, stack]
    id_activated = np.where(chosen_names_subset == true_name)[0][0]
    activated_normal = normal(activated_mu, activated_sigma)
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
    
    P = np.clip(sim_table_subset[id_activated] * activated_normal + normal(np.zeros(l),noise_sigma * np.ones(l)), 0, 1)

    if regulation_policy == 'fake_arity_decidable_wrt_true': # All templates which can be recognized has activate type
        chosen_names_subset_ = get_templates_decisive_based_on_arity(names, true_name, min_ch, scene)
        
        for chosen_name_subset_ in chosen_names_subset_:
            if chosen_name_subset_ not in chosen_names_subset:
                chosen_names_subset = np.append(chosen_names_subset,chosen_name_subset_)
                P = np.append(P, activated_normal)
            else:
                id = list(chosen_names_subset).index(chosen_name_subset_)
                P[id] = np.clip(activated_normal + normal(0, noise_sigma), 0,1)

    elif regulation_policy == 'undecidable_wrt_true':
        # 1. choose random subset 
        
        chosen_names_subset_1 = get_templates_decisive_based_on_arity(names, true_name, min_ch, scene)
        chosen_names_subset_2 = get_templates_decisive_based_on_properties(names, true_name, min_ch, scene)
        chosen_names_subset_ = list(set(chosen_names_subset_1 + chosen_names_subset_2))

        for chosen_name_subset_ in chosen_names_subset_:
            if chosen_name_subset_ not in chosen_names_subset:
                chosen_names_subset = np.append(chosen_names_subset,chosen_name_subset_)
                P = np.append(P, activated_normal)
            else:
                id = list(chosen_names_subset).index(chosen_name_subset_)
                P[id] = np.clip(activated_normal + normal(0, noise_sigma),0,1)

    if regulation_policy == 'fake_properties_decidable_wrt_true':
        chosen_names_subset_ = get_templates_decisive_based_on_properties(names, true_name, 
        min_ch, scene)
        
        for chosen_name_subset_ in chosen_names_subset_:
            if chosen_name_subset_ not in chosen_names_subset:
                chosen_names_subset = np.append(chosen_names_subset,chosen_name_subset_)
                P = np.append(P, activated_normal)
            else:
                id = list(chosen_names_subset).index(chosen_name_subset_)
                P[id] = np.clip(activated_normal + normal(0, noise_sigma),0,1)


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
    t_cts = {}
    for tt in scene.templates:
        t_cts[tt.name] = tt.compare_types
    
    chosen_names_subset = []
    for k in t_cts.keys():
        if t_cts[k] != t.compare_types:
            chosen_names_subset.append(k)    
    
    if true_name not in chosen_names_subset: chosen_names_subset.append(true_name)
    # 2. assign probs
    #act_p = normal(gen_params['activated_prob_mu'], gen_params['activated_prob_sigma'])
    #P = act_p * np.ones(len(chosen_names_subset))
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
        is_unfeasible_template = False
        for o in scene.selections:
            for s in scene.storages:
                if not template.is_feasible(o, s):
                    is_unfeasible_template = True
        
        if is_unfeasible_template:
            unfeasible_templates.append(template.name)


    return unfeasible_templates