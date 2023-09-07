class PushTask():
    def __init__(self):
        self.name = 'push'
        self.compare_types = ['template', 'selections']
        self.complexity = 1

    def has_compare_type(self, compare_type):
        if compare_type in self.compare_types:
            return True
        else:
            return False

    def task_property_penalization(self, property):
        ''' How much to penalize for given property - weighted
            Set up using common sense
            e.g. when object is not reachable, how much it matters for pick-task -> quite significant
        '''
        return {'reachable': 0.0,
                'pickable':  0.8, 
                'stackable': 1.0,
                'pushable':  0.0, 
                'full':      0.9,
                'glued':     1.0,
            }[property]

    def is_feasible(self, o, s=None):

        if (o.properties['reachable'] and
            o.properties['pushable'] and
            not o.properties['glued']
            ):
            return True
        else:
            return False