#!/usr/bin/env python3

import logging

class UnglueTask():
    def __init__(self):
        self.name = 'unglue'
        self.compare_types = ['template', 'selections']
        self.complexity = 1

    def has_compare_type(self, compare_type):
        if compare_type in self.compare_types:
            return True
        else:
            return False

    def task_property_penalization_selections(self, property):
        ''' How much to penalize for given property - weighted
            Set up using common sense
            e.g. when object is not reachable, how much it matters for pick-task -> quite significant
        '''
        return {'reachable': 1.0,
                'pickable':  1.0, 
                'stackable': 1.0,
                'pushable':  1.0, 
                'full-stack':1.0,
                'full-liquid':1.0,
                'glued':     0.0,
            }[property]

    def is_feasible(self, o=None, s=None):
        #assert s is None
        assert o is not None

        if ( o.properties['glued'] ):
            return True
        else:
            return False