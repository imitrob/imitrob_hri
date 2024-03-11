import yaml
import numpy as np
import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from merging_modalities.utils import cc
from imitrob_hri.imitrob_nlp.TemplateFactory import create_template
from copy import deepcopy
from merging_modalities import noise_model
from pathlib import Path

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
            'types': ['liquid-container', 'object', 'container'],
            'glued': False
        }, properties_config=None):
        
        self.name = observations['name']

        self.properties = {}
        if properties_config is not None:
            self.observations = None

            for k, v in properties_config.items():
                if type(v) is bool:
                    self.properties[k] = lambda value=v: (int(value), value)
                else:
                    self.properties[k] = v
        else:
            assert 'types' in observations
            self.observations = observations

            self.properties['glued'] = self.glued
            # if 'object' in observations['types']:
            self.properties['pickable'] = self.pickable
            self.properties['reachable'] = self.reachable
            self.properties['stackable'] = self.stackable
            self.properties['pushable'] = self.pushable
            # if 'liquid-container' in observations['types']:
            self.properties['full-stack'] = self.full_stack
            self.properties['full-liquid'] = self.full_liquid
            self.properties['full-container'] = self.full_container
            if len(observations['types']) > 0:
                self.properties['type'] = self.observations['types'][0]
            else:
                self.properties['type'] = []

    def is_type(self, typ):
        #print("is type::!: ", typ, self.properties[typ](), typ in self.properties)
        return typ in self.properties and self.properties[typ]()[1] or typ == self.properties["type"]
    

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
        eval = (self.observations['roundness-top'] >= threshold_roundtop_being_unstackable)
        return round(penalization,r), eval

    def pushable(self, r=4):
        threshold_weight_being_unpushable = 2 # [kg]
        
        penalization = self.sigmoid(self.observations['weight'], center=threshold_weight_being_unpushable)
        eval = (self.observations['weight'] <= threshold_weight_being_unpushable)
        return round(penalization,r), eval

    def full_stack(self, r=4):
        return round(int(self.observations['contain_item']),r), int(self.observations['contain_item'])
    
    def full_container(self, r=4):
        return round(int(self.observations['contain_item']),r), int(self.observations['contain_item'])

    def full_liquid(self, r=4):
        threshold_capacity_being_full = 0.7 # [norm. rate being full]
        
        penalization = self.isigmoid(self.observations['contains'], center=threshold_capacity_being_full)
        eval = (self.observations['contains'] >= threshold_capacity_being_full)
        return round(penalization,r), eval     

    def glued(self, r=4):
        return round(int(self.observations['glued']),r), int(self.observations['glued'])
    
    ''' Modelling functions: '''
    @staticmethod
    def gaussian(x, sigma=0.2):
        return np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))

    @staticmethod
    def isigmoid(x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + np.exp((center-x)*(tau)))

    @staticmethod
    def sigmoid(x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + np.exp((center-x)*(-tau)))
    
    def __str__(self):
        if self.observations is None:
            return f"{cc.F}{self.name}{cc.E}: properties loaded from config\n"
        return f"{cc.F}{self.name}{cc.E}: s: {round(self.observations['size'],2)}, p: {np.round(self.observations['position'],2)}, roundness: {round(self.observations['roundness-top'],2)}, weight: {round(self.observations['weight'],2)}, contains: {round(self.observations['contains'],2)}, glued: {self.observations['glued']}, types: {self.observations['types']}\nProperties: {cc.H}Glued:{cc.E} {self.glued(2)}, {cc.H}Pickable:{cc.E} {self.pickable(2)}, {cc.H}Reachable:{cc.E} {self.reachable(2)}, {cc.H}Stackable {cc.E}{self.stackable(2)}, {cc.H}Pushable:{cc.E} {self.pushable(2)}, {cc.H}Full stack:{cc.E} {self.full_stack(2)} full liquid: {self.full_liquid(2)}\n"
    
    def get_all_properties(self):
        s = ""
        for k, v in self.properties.items():
            if type(v) is str:
                s += f"{k}: {v}\n"
            else:
                s += f"{k}: {v()[1]}\n"
        return s
    
    
class Scene3():
    def __init__(self, selections, storages, template_names):
        '''
        Created templates are all templates from set, used only as info about classes
        '''
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


def get_n_objects(c):
    # Based on configuration, choose how many objects and storages to generate in the scene
    if c.scene_gen_config['selections_n'][0] == 'uniform':
        low, high = c.scene_gen_config['selections_n'][1:3]
        return np.random.randint(low, high)
    else: raise Exception("TODO ADD OTHER THAN UNIFORM PROB DISTR.")
    
def get_n_storages(c):
    # Based on configuration, choose how many objects and storages to generate in the scene
    if c.scene_gen_config['storages_n'][0] == 'uniform':
        low, high = c.scene_gen_config['storages_n'][1:3]
        return np.random.randint(low, high)
    else: raise Exception("TODO ADD OTHER THAN UNIFORM PROB DISTR.")
  

def get_random_scene(c, object_name_list=['potted meat can', 'tomato soup can', 'bowl', 'box', 'big box', 'paper', 'wrench', 'glued wrench'], storage_name_list=['paper box', 'abstract marked zone', 'out of table', 'on the table']):
    object_name_list = c.mm_pars_names_dict['selections']
    storage_name_list = c.mm_pars_names_dict['storages']

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
            'types': ['liquid-container', 'object', 'container'],
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
            'types': ['liquid-container', 'object', 'container'],
            'glued': np.random.randint(0,2),
        }
        storages.append(Object3(observations))

    scene = Scene3(objects, storages, template_names=c.templates)

    return scene





def create_scene_from_fake_data(exp=None, run=None):

    if exp is None and run is None:
        with open(Path("~/crow-base/config/crow_hri/scene_properties.yaml").expanduser()) as f:
            scene_properties = yaml.safe_load(f)
    else:
        with open(Path("~/crow-base/config/crow_hri/all_scene_properties.yaml").expanduser()) as f:
            all_scene_properties = yaml.safe_load(f)
            scene_properties = all_scene_properties[exp][int(run)]

    scene_objects = []
    object_names = []
    scene_storages = []
    storage_names = []
    for name in scene_properties.keys():
        
        o = Object3(observations={'name': name}, properties_config=scene_properties[name])

        if name not in object_names and name not in storage_names:
            if o.is_type("object"):
                object_names.append(name)
                scene_objects.append(o)
            elif o.is_type("storage"):
                storage_names.append(name)
                scene_storages.append(o)
            else:
                raise ValueError(f"Unknown object type for object {o}")
    
    s = None
    template_names = ['pick up', 'point']
    s = Scene3(scene_objects, scene_storages, template_names)
    return s

if __name__ == '__main__':
    s = create_scene_from_fake_data()
    print("s", s)