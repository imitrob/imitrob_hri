''' Default configuration class '''
from abc import ABC

class Configuration(ABC):

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
        self.match_threshold = 0.55
        self.clear_threshold = 0.5
        self.unsure_threshold = 0.15
        self.diffs_threshold = 0.01

        self.epsilon = 0.9
        self.gamma = 0.5

        self.DEBUG = False

class Configuration2(Configuration):
    def __init__(self):
        self.ct_names = {'template': ['move up', 'release', 'stop',
            'pick', 'place', 'push', 'point',
            ],
            'selections': ['potted meat can', 'tomato soup can', 'bowl', 'box', 'big box', 'paper', 'wrench', 'glued wrench'],
            'storages': ['paper box', 'abstract marked zone', 'out of table', 'on the table'],
        }
        self.match_threshold = 0.55
        self.clear_threshold = 0.5
        self.unsure_threshold = 0.15
        self.diffs_threshold = 0.01

        self.epsilon = 0.9
        self.gamma = 0.5

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
        self.match_threshold = 0.55
        self.clear_threshold = 0.5
        self.unsure_threshold = 0.15
        self.diffs_threshold = 0.01

        self.epsilon = 0.9
        self.gamma = 0.5

        self.DEBUG = False