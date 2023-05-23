
def init():
    global action_names, object_names, match_threshold, clear_threshold, unsure_threshold, diffs_threshold
    action_names = ['pick up', 'place', 'push']
    object_names = ['box', 'big box', 'table']
    match_threshold = 0.5
    clear_threshold = 0.4
    unsure_threshold = 0.2
    diffs_threshold = 0.05