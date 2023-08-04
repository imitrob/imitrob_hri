
def init():
    global action_names, object_names, match_threshold, clear_threshold, unsure_threshold, diffs_threshold, compare_types, object_properties, selection_penalization
    #action_names = ['pick up', 'place', 'push']
    action_names = ['PickTask', 'PointTask']
    object_names = ['box', 'big box', 'table']
    compare_types = ['action', 'selection']
    match_threshold = 0.25
    clear_threshold = 0.2
    unsure_threshold = 0.15
    diffs_threshold = 0.05

    object_properties = {
        'box': {
            'reachable': True,
            'pickable': True,
        }, 'big box': {
            'reachable': True,
            'pickable': False,
        }, 'table': {
            'reachable': True ,
            'pickable': False,
        },
    }

    selection_penalization = {
        'PickTask': {
            'reachable': 0.8,
            'pickable': 0.0,
        }, 'PointTask': {
            'reachable': 1.0,
            'pickable': 1.0,
        },
    }