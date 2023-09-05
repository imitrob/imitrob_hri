class PlaceTask():
    def __init__(self):
        self.compare_types = ['action', 'storage']
        self.complexity = 1

    def has_compare_type(self, compare_type):
        return False


    def has_compare_type(self, compare_type):
        if compare_type in self.compare_types:
            return True
        else:
            return False