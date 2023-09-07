''' Default configuration class '''
from abc import ABC

class Configuration(ABC):
    @property
    def ct_properties_default(self):
        return {
            'template': [],
            'selections': ['pickable', 'reachable', 'stackable', 'pushable', 'full', 'glued'],
            'storages': ['pickable', 'reachable', 'stackable', 'pushable', 'full', 'glued'],
        }
    
    @property
    def scene_gen_config_default(self):
        return {
            'selections_n': ('uniform', 1, 4),
            'storages_n': ('uniform', 0, 3),
        }
    
    @property
    def match_threshold_default(self):
        return 0.55
    @property
    def clear_threshold_default(self):
        return 0.50
    @property
    def unsure_threshold_default(self):
        return 0.15
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
        return self.ct_names['template']

    @property
    def selections(self):
        return self.ct_names['selections']

    @property
    def storages(self):
        return self.ct_names['storages']

class Configuration1(Configuration):
    def __init__(self):
        self.ct_names = {'template': ['move up', 'release', 'stop', ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'big box', 'paper', 'wrench', 'glued wrench'],
            'storages': [],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 1, 3),
            'storages_n': ('uniform', 0, 1),
        }
        self.ct_properties = self.ct_properties_default

        self.match_threshold = self.match_threshold_default
        self.clear_threshold = self.clear_threshold_default
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = self.alpha_penal_default

        self.DEBUG = False

class Configuration2(Configuration):
    def __init__(self):
        self.ct_names = {'template': ['move up', 'release', 'stop',
            'pick', 'place', 'push', 'point',
            ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'big box', 'paper', 'wrench', 'glued wrench'],
            'storages': ['paper box', 'abstract marked zone', 'out of table', 'on the table'],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 2, 5),
            'storages_n': ('uniform', 0, 2),
        }
        self.ct_properties = self.ct_properties_default

        self.match_threshold = self.match_threshold_default
        self.clear_threshold = self.clear_threshold_default
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = self.alpha_penal_default

        self.DEBUG = False

class Configuration3(Configuration):
    def __init__(self):
        self.ct_names = {'template': ['move up', 'release', 'stop',
                'pick', 'place', 'push', 'point',
                'pour', 'put',
                ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'big box', 'paper', 'wrench', 'glued wrench'],
            'storages': ['paper box', 'abstract marked zone', 'out of table', 'on the table'],
        }
        self.scene_gen_config = {
            'selections_n': ('uniform', 3, 7),
            'storages_n': ('uniform', 2, 3),
        }
        self.ct_properties = self.ct_properties_default
        
        self.match_threshold = self.match_threshold_default
        self.clear_threshold = self.clear_threshold_default
        self.unsure_threshold = self.unsure_threshold_default
        self.diffs_threshold = self.diffs_threshold_default

        self.epsilon = self.epsilon_default
        self.gamma = self.gamma_default
        self.alpha_penal = self.alpha_penal_default

        self.DEBUG = False