import numpy as np
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from nlp_new.templates.MoveUpTask import MoveUpTask
from nlp_new.templates.PickTask import PickTask
from nlp_new.templates.PlaceTask import PlaceTask
from nlp_new.templates.PointTask import PointTask
from nlp_new.templates.PourTask import PourTask
from nlp_new.templates.PushTask import PushTask
from nlp_new.templates.PutTask import PutTask
from nlp_new.templates.ReleaseTask import ReleaseTask
from nlp_new.templates.StopTask import StopTask
from merging_modalities.utils import cc

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
            self.properties['full'] = self.full


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

    def full(self, r=4):
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
        return f"{cc.F}{self.name}{cc.E}: s: {round(self.observations['size'],2)}, p: {np.round(self.observations['position'],2)}, roundness: {round(self.observations['roundness-top'],2)}, weight: {round(self.observations['weight'],2)}, contains: {round(self.observations['contains'],2)}, glued: {self.observations['glued']}\nProperties: {cc.H}Glued:{cc.E} {self.glued(2)}, {cc.H}Pickable:{cc.E} {self.pickable(2)}, {cc.H}Reachable:{cc.E} {self.reachable(2)}, {cc.H}Stackable {cc.E}{self.stackable(2)}, {cc.H}Pushable:{cc.E} {self.pushable(2)}, {cc.H}Full:{cc.E} {self.full(2)}\n"
    
    
class Scene3():
    def __init__(self, selections, storages):
        self.templates = [
            MoveUpTask(),
            PickTask(),
            PlaceTask(),
            PointTask(),
            PourTask(),
            PushTask(),
            PutTask(),
            ReleaseTask(),
            StopTask(),
        ]
        self.selections = selections
        self.storages = storages

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
            'types': ['liquid container', 'object'],
            'glued': False,
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
            'types': ['liquid container', 'object'],
            'glued': False,
        }
        storages.append(Object3(observations))

    scene = Scene3(objects, storages)

    return scene


def get_random_feasible_triplet(scene):
    template = None
    arange = list(range(len(scene.templates)))
    np.random.shuffle(arange)
    for t_n in arange:
        t = scene.templates[t_n]
        #if t.is_feasible(): # possibility to restrict template based on scene config
        template = t
        break
    
    requirements = template.compare_types # requirements to fill
    
    if 'template' in requirements:
        requirements.remove('template')

    object = None
    if 'selections' in requirements:
        # choose an object  that is feasible 
        arange = list(range(len(scene.selections)))
        np.random.shuffle(arange)
        for o_n in arange:
            o = scene.selections[o_n]
            if t.is_feasible(o):
                object = o
        requirements.remove('selections')

    storage = None
    if 'storages' in requirements:
        # choose an storage that fits
        arange = list(range(len(scene.storages)))
        np.random.shuffle(arange)
        for s_n in arange:
            s = scene.storages[s_n]
            if t.is_feasible(object, s):
                storage = s
        requirements.remove('storages')

    if len(requirements) > 0:
        raise Exception(f"TODO HERE TO ADD NEW COMPARE TYPE {requirements}")

    return triplet_to_name(template, object, storage)

def triplet_to_name(template, object, storage):
    if template is not None:
        template = template.name
    if object is not None:
        object = object.name
    if storage is not None:
        storage = storage.name
    return template, object, storage


def get_random_names_and_likelihoods(names, true_name, gen_params, min_ch):
    activated_mu = gen_params['activated_prob_mu']
    activated_sigma = gen_params['activated_prob_sigma']
    unsure_mu = gen_params['unsure_prob_mu']
    unsure_sigma = gen_params['unsure_prob_sigma']

    if len(names) == 0 or true_name is None: 
        return np.array([]), np.array([])
    # Get observed gesture templates
    ct_template_gesture_chosen_names = np.random.choice(names, size=np.random.randint(min_ch, len(names) ), replace=False)
    ct_template_gesture_chosen_names = add_if_not_there(ct_template_gesture_chosen_names, true_name)
    # Give all likelihoods unsure probs.
    ct_template_gesture_likelihood = [normal(unsure_mu, unsure_sigma) for _ in range(len(ct_template_gesture_chosen_names))]
    # Give the true value the activated prob
    ct_template_gesture_likelihood[np.where(ct_template_gesture_chosen_names == true_name)[0][0]] = normal(activated_mu, activated_sigma)

    return ct_template_gesture_likelihood, ct_template_gesture_chosen_names

