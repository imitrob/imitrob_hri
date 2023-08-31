''' Default configuration class '''
class Configuration():
    def __init__(self):
        self.ct_names = {
            'template': [],
            'selections': [],
        }
        self.compare_types = []

        self.match_threshold = None
        self.clear_threshold = None
        self.unsure_threshold = None
        self.diffs_threshold = None

        self.object_properties = {}

        self.task_property_penalization = {}
        self.DEBUG = False
