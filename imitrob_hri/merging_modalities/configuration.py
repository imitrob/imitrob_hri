''' Default configuration class '''
from abc import ABC
import numpy as np

class Configuration(ABC):
    @property
    def ct_properties_default(self):
        return {
            'template': [],
            'selections': ['pickable', 'reachable', 'stackable', 'pushable', 'full-liquid', 'full-stack', 'glued'],
            'storages': ['pickable', 'reachable', 'stackable', 'pushable', 'full-liquid', 'full-stack', 'glued'],
        }
    
    @property
    def scene_gen_config_default(self):
        return {
            'selections_n': ('uniform', 1, 4),
            'storages_n': ('uniform', 0, 3),
        }

    @property
    def match_threshold_default(self):
        return 0.25
    @property
    def clear_threshold_default(self):
        return 0.2
    @property
    def unsure_threshold_default(self):
        return 0.11
    @property
    def diffs_threshold_default(self):
        return 0.01
    
    @property
    def epsilon_default(self):
        return 0.9
    @property
    def gamma_default(self):
        return 0.5
    @property
    def alpha_penal_default(self):
        return 0.9
        
    @property
    def templates(self):
        return self.mm_pars_names_dict['template']

    @property
    def selections(self):
        return self.mm_pars_names_dict['selections']
    @property
    def objects(self):
        return self.mm_pars_names_dict['selections']

    @property
    def storages(self):
        return self.mm_pars_names_dict['storages']
    
    @property
    def sim_table_gesture_default(self):

        return np.array([
            [1,	    0.01,	0.003,	0.032,	0,	    0.446,	0,	    0.014, 0.002],
            [0.001,	1,  	0,	    0.001,	0.002,	0.002,	0,	    0,     0.   ],
            [0, 	0,  	1,	    0,      0.011,	0.007,	0.603,	0,     0.014],
            [0.005,	0.002,	0,	    1,	    0.013,	0.01,	0,	    0,     0.002],
            [0, 	0.016,	0.017,	0.002,	1,	    0,	    0.002,	0.042, 0.009],
            [0.37,	0.001,	0.049,	0.026,	0.001,	1,	    0.019,	0,     0.4  ],
            [0.018,	0.001,	0,	    0.6,	0.01,	0.015,	1,	    0,     0.020],
            [0.006,	0.002,	0,	    0.6,	0.016,	0.011,	0,	    1,     0.   ],
            [0.002,	0.001,	0,	    0.01,	0.01,	0.005,	0,	    0,     1    ],
        ])

    @property
    def sim_table_language_default(self):
        return np.array([
            [ 1.  , 0.  , 0.33, 0.4 , 0.33, 0.40, 0.33, 0.4 , 0.4 ],
            [ 0.  , 1.  , 0.33, 0.  , 0.67, 0.  , 0.  , 0.4 , 0.  ],
            [ 0.33, 0.33, 1.  , 0.4 , 0.33, 0.40, 0.33, 0.40, 0.40],
            [ 0.40, 0.  , 0.4 , 1.  , 0.4 , 0.5 , 0.4 , 0.5 , 0.5 ],
            [ 0.33, 0.67, 0.33, 0.40, 1.  , 0.4 , 0.33, 0.40, 0.4 ],
            [ 0.40, 0.  , 0.40, 0.50, 0.4 , 1.  , 0.4 , 0.5 , 0.5 ],
            [ 0.33, 0.  , 0.33, 0.40, 0.33, 0.4 , 1.  , 0.4 , 0.8 ],
            [ 0.4 , 0.4 , 0.4 , 0.5 , 0.4 , 0.5 , 0.4 , 1.  , 0.5 ],
            [ 0.4 , 0.  , 0.4 , 0.5 , 0.4 , 0.5 , 0.8 , 0.5 , 1.  ],
        ])

    @property
    def sim_table_language_objects_default(self):
        #['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench']
        return np.array([
            [1.  , 0.71, 0.22, 0.4 , 0.33, 0.2 , 0.2 ],
            [0.71, 1.  , 0.22, 0.4 , 0.33, 0.2 , 0.2 ],
            [0.22, 0.22, 1.  , 0.4 , 0.29, 0.4 , 0.  ],
            [0.4 , 0.4 , 0.4 , 1.  , 0.75, 0.33, 0.  ],
            [0.33, 0.33, 0.29, 0.75, 1.  , 0.5 , 0.  ],
            [0.2 , 0.2 , 0.4 , 0.33, 0.5 , 1.  , 0.33],
            [0.2 , 0.2 , 0.  , 0.  , 0.  , 0.33, 1.  ],
       ])

    @property
    def sim_table_gesture_objects_default(self):
        #['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench']
        # TODO
        return np.array([
            [1.  , 0.22, 0.22, 0.22, 0.22, 0.22, 0.22],
            [0.22, 1.  , 0.22, 0.22, 0.22, 0.22, 0.22],
            [0.22, 0.22, 1.  , 0.22, 0.22, 0.22, 0.22],
            [0.22, 0.22 ,0.22, 1.  ,0.22, 0.22, 0.22],
            [0.22, 0.22, 0.22, 0.22, 1.  , 0.22, 0.22],
            [0.22, 0.22, 0.22, 0.22, 0.22, 1.  , 0.22],
            [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 1.  ],
       ])

    @property
    def samples_default(self):
        return 10000
    
    ''' This is list of Observations and Properties is fixed, cannot be added, they are tied to Object3 definition '''
    @property
    def observations(self):
        return ['name', 'size', 'position', 'roundness-top', 'weight', 'contains', 'contain_item', 'glued']
    
    @property
    def properties(self):
        return ['glued', 'pickable', 'reachable', 'stackable', 'pushable', 'full-stack', 'full-liquid']
    
class Configuration1(Configuration):
    def __init__(self):
        self.mm_pars_names_dict = {'template': ['move-up', 'release', 'stop', ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench'],
            'storages': [],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 1, 3),
            'storages_n': ('uniform', 0, 1),
        }
        #self.ct_properties = self.ct_properties_default

        self.match_threshold = 0.4
        self.clear_threshold = 0.4
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = 0.0

        self.DEBUG = False

        self.sim_table_gesture = self.sim_table_gesture_default[0:3,0:3]
        self.sim_table_language = self.sim_table_language_default[0:3,0:3]

        self.sim_table_language_objects = self.sim_table_language_objects_default
        self.sim_table_gesture_objects = self.sim_table_gesture_objects_default

        self.sim_table_language_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO
        self.sim_table_gesture_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO

        self.samples = self.samples_default


class Configuration2(Configuration):
    def __init__(self):
        self.mm_pars_names_dict = {'template': ['move-up', 'release', 'stop',
                                    'pick', 'push', 'unglue',
            ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench'],
            'storages': ['paper box', 'abstract marked zone', 'out of table', 'on the table'],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 2, 5),
            'storages_n': ('uniform', 0, 2),
        }
        #self.ct_properties = self.ct_properties_default

        self.match_threshold = 0.4
        self.clear_threshold = 0.4
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = 0.1

        self.DEBUG = False

        self.sim_table_gesture = self.sim_table_gesture_default[0:6,0:6]
        self.sim_table_language = self.sim_table_language_default[0:6,0:6]

        self.sim_table_language_objects = self.sim_table_language_objects_default
        self.sim_table_gesture_objects = self.sim_table_gesture_objects_default

        self.sim_table_language_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO
        self.sim_table_gesture_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO

        self.samples = self.samples_default

class Configuration3(Configuration):
    def __init__(self):
        self.mm_pars_names_dict = {'template': ['move-up', 'release', 'stop',
                                    'pick', 'push', 'unglue',
                                    'pour', 'put-into', 'stack',
                ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench'],
            'storages': ['paper box', 'abstract marked zone', 'out of table', 'on the table'],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 3, 7),
            'storages_n': ('uniform', 2, 3),
        }
        #self.ct_properties = self.ct_properties_default
        
        self.match_threshold = self.match_threshold_default
        self.clear_threshold = self.clear_threshold_default
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = 0.2

        self.DEBUG = False

        self.sim_table_gesture = self.sim_table_gesture_default
        self.sim_table_language = self.sim_table_language_default

        self.sim_table_language_objects = self.sim_table_language_objects_default
        self.sim_table_gesture_objects = self.sim_table_gesture_objects_default

        self.sim_table_language_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO
        self.sim_table_gesture_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO

        self.samples = self.samples_default



''' Crow '''

class ConfigurationCrow1(Configuration):
    def __init__(self):
        # need to be the default names
        self.mm_pars_names_dict = {'template': ['pick', 'point', 'pass','release'],
            'selections': ['cube holes'],
            'storages': [],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 3, 7),
            'storages_n': ('uniform', 2, 3),
        }
        #self.ct_properties = self.ct_properties_default
        
        self.match_threshold = self.match_threshold_default
        self.clear_threshold = self.clear_threshold_default
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = 0.2

        self.DEBUG = False

        self.sim_table_gesture = self.sim_table_gesture_default
        self.sim_table_language = self.sim_table_language_default

        self.sim_table_language_objects = self.sim_table_language_objects_default
        self.sim_table_gesture_objects = self.sim_table_gesture_objects_default

        self.sim_table_language_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO
        self.sim_table_gesture_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO

        self.samples = self.samples_default

ConfigurationDefault = ConfigurationCrow1


class Configuration2_1(Configuration):
    def __init__(self):
        self.mm_pars_names_dict = {'template': ['move-up', 'release', 'stop', ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench'],
            'storages': [],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 0, 1), # No Objects
            'storages_n': ('uniform', 0, 1), # No Storages
        }
        #self.ct_properties = self.ct_properties_default

        self.match_threshold = 0.4
        self.clear_threshold = 0.4
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = 0.0

        self.DEBUG = False

        self.sim_table_gesture = self.sim_table_gesture_default[0:3,0:3]
        self.sim_table_language = self.sim_table_language_default[0:3,0:3]

        self.sim_table_language_objects = self.sim_table_language_objects_default
        self.sim_table_gesture_objects = self.sim_table_gesture_objects_default

        self.sim_table_language_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO
        self.sim_table_gesture_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO

        self.samples = self.samples_default


class Configuration2_2(Configuration):
    def __init__(self):
        self.mm_pars_names_dict = {'template': ['move-up', 'release', 'stop',
                                    'pick', 'push', 'unglue',
            ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench'],
            'storages': ['paper box', 'abstract marked zone', 'out of table', 'on the table'],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 2, 3), # Always two objects
            'storages_n': ('uniform', 1, 2), # Always one storage 
        }
        #self.ct_properties = self.ct_properties_default

        self.match_threshold = 0.4
        self.clear_threshold = 0.4
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = 0.1

        self.DEBUG = False

        self.sim_table_gesture = self.sim_table_gesture_default[0:6,0:6]
        self.sim_table_language = self.sim_table_language_default[0:6,0:6]

        self.sim_table_language_objects = self.sim_table_language_objects_default
        self.sim_table_gesture_objects = self.sim_table_gesture_objects_default

        self.sim_table_language_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO
        self.sim_table_gesture_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO

        self.samples = self.samples_default

class Configuration2_3(Configuration):
    def __init__(self):
        self.mm_pars_names_dict = {'template': ['move-up', 'release', 'stop',
                                    'pick', 'push', 'unglue',
                                    'pour', 'put-into', 'stack',
                ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'alt box', 'paper', 'wrench'],
            'storages': ['paper box', 'abstract marked zone', 'out of table', 'on the table'],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 6, 7), # Always 6 Objects
            'storages_n': ('uniform', 3, 4), # Always 3 Storages
        }
        #self.ct_properties = self.ct_properties_default
        
        self.match_threshold = self.match_threshold_default
        self.clear_threshold = self.clear_threshold_default
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = 0.2

        self.DEBUG = False

        self.sim_table_gesture = self.sim_table_gesture_default
        self.sim_table_language = self.sim_table_language_default

        self.sim_table_language_objects = self.sim_table_language_objects_default
        self.sim_table_gesture_objects = self.sim_table_gesture_objects_default

        self.sim_table_language_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO
        self.sim_table_gesture_storages = self.sim_table_gesture_objects_default[0:4,0:4] # TODO

        self.samples = self.samples_default
