
def init():
    global template_names, selection_names, match_threshold, clear_threshold, unsure_threshold
    global diffs_threshold, compare_types, object_properties, task_property_penalization
    #template_names = ['pick up', 'place', 'push']
    template_names = ['PickTask', 'PointTask', 'PutTask']
    selection_names = ['box', 'big box', 'table']
    compare_types = ['template', 'selections']

    match_threshold = 0.48
    clear_threshold = 0.44
    unsure_threshold = 0.03
    diffs_threshold = 0.001

    object_properties = {
        'box': {
            'reachable': True,
            'pickable': True,
        },
        'big box': {
            'reachable': True,
            'pickable': False,
        },
        'table': {
            'reachable': True ,
            'pickable': False,
        },


        'Cube': {
            'reachable': True,
            'pickable': True,
        },
        'Peg': {
            'reachable': True,
            'pickable': True,
        },
        'aruco box': {
            'reachable': True,
            'pickable': True,
        },
        'Cup': {
            'reachable': True,
            'pickable': True,
        },
                


    }

    task_property_penalization = {
        'PickTask': {
            'reachable': 0.8,
            'pickable': 0.0,
        }, 'PointTask': {
            'reachable': 1.0,
            'pickable': 1.0,
        },
    }

    global DEBUG
    DEBUG = False