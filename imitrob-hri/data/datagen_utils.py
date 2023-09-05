import numpy as np

def normal(mu, sigma):
    ''' normal cropped (0 to 1) '''
    return np.clip(np.random.normal(mu, sigma), 0, 1)


def add_if_not_there(a, b):
    if b not in a:
        return np.append(a, b)
    return a


class Object3():
    def __init__(self, observations = {
            'name': 'box',
            'size': 0.1, # [m]
            'position': [0.0, 0.0, 0.0], # [m,m,m]
            'roundness-top': 0.1, # [normalized belief rate]
            'weight': 1, # [kg]
            'contains': 0.1, # normalized rate being full 
            'types': ['liquid container', 'object'],
            'glued': False
        }):

        self.observations = observations

        self.properties = {}
        self.properties['glued'] = self.glued
        if type == 'object':
            self.properties['pickable'] = self.pickable
            self.properties['reachable'] = self.reachable
            self.properties['stackable'] = self.stackable
            self.properties['pushable'] = self.pushable
        elif type == 'liquid container':
            self.properties['full'] = self.full


    def pickable(self):
        threshold_being_unpickable = 0.12 # [m] <- robot's gripper max opened distance 

        penalization = self.sigmoid(self.observations['size'], center=threshold_being_unpickable)
        eval = self.observations['size'] <= threshold_being_unpickable
        return penalization, eval

    def reachable(self):
        threshold_radius_being_unreachable = 0.6 # [m]
        penalization, eval = 1.0, True
        for x in self.observations['position']:
            p = self.sigmoid(x, center=threshold_radius_being_unreachable)
            penalization *= p
            e = (x <= threshold_radius_being_unreachable)
            eval &= e
        return penalization, eval
    
    def stackable(self):
        ''' Depending on the top surface of the object '''
        threshold_roundtop_being_unstackable = 0.1 # [belief rate]
        
        penalization = self.sigmoid(self.observations['roundness-top'], center=threshold_roundtop_being_unstackable)
        eval = (self.observations['roundness-top'] <= threshold_roundtop_being_unstackable)
        return penalization, eval

    def pushable(self):
        threshold_weight_being_unpushable = 2 # [kg]
        
        penalization = self.sigmoid(self.observations['weight'], center=threshold_weight_being_unpushable)
        eval = (self.observations['weight'] <= threshold_weight_being_unpushable)
        return penalization, eval

    def full(self):
        threshold_capacity_being_full = 0.7 # [norm. rate being full]
        
        penalization = self.sigmoid(self.observations['contains'], center=threshold_capacity_being_full)
        eval = (self.observations['contains'] <= threshold_capacity_being_full)
        return penalization, eval     
    
    def glued(self):
        return self.observations['glued']
    
    ''' Modelling functions: '''
    @staticmethod
    def gaussian(x, sigma=0.2):
        return np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))

    @staticmethod
    def sigmoid(x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + np.exp((center-x)*(-tau)))
    

def get_random_scene(y_selection, gen_params, object_name_list=['potted meat can', 'tomato soup can', 'bowl', 'box', 'big box', 'paper', 'wrench', 'glued wrench']):
    nobjs = np.random.randint(1,3)
    objects = []

    object_chosen_names = list(np.random.choice(object_name_list, size=nobjs, replace=False))
    if y_selection not in object_chosen_names:
        object_chosen_names.append(y_selection)

    for object_name in object_chosen_names:
        observations = {
            'name': object_name, 
            'size': np.random.random() * 0.5, # [m]
            'position': [np.random.random(), np.random.random(), 0.0], # [m,m,m]
            'roundness-top': np.random.random() * 0.2, # [normalized belief rate]
            'weight': np.random.random() * 4, # [kg]
            'contains': np.random.random(), # normalized rate being full 
            'types': np.random.choice(['liquid container', 'object'], size=np.random.randint(0,2), replace=False)
        }

        if gen_params['only_single_object_action_performable'] and object_name != y_selection:
            ### Assign unsatisfyable observations
            observations['size'] += 1 # [m]
            observations['position'] += [1.0, 1.0, 1.0] # [m,m,m]
            observations['roundness-top'] += 0.3 # [normalized belief rate]
            observations['weight'] += 5 # [kg]
            observations['contains'] = 1 # normalized rate being full 
            observations['glued'] = True
            


        objects.append(Object3(observations))
    return objects





def get_random_names_and_likelihoods(names, true_name, gen_params, min_ch):
    activated_mu = gen_params['activated_prob_mu']
    activated_sigma = gen_params['activated_prob_sigma']
    unsure_mu = gen_params['unsure_prob_mu']
    unsure_sigma = gen_params['unsure_prob_sigma']

    if len(names) == 0 or true_name is None: 
        return np.array([]), np.array([])
    # Get observed gesture templates
    ct_template_gesture_chosen_names = np.random.choice(names, size=np.random.randint(min_ch, len(names) ), replace=False)
    ct_template_gesture_chosen_names = add_if_not_there(ct_template_gesture_chosen_names, true_name)
    # Give all likelihoods unsure probs.
    ct_template_gesture_likelihood = [normal(unsure_mu, unsure_sigma) for _ in range(len(ct_template_gesture_chosen_names))]
    # Give the true value the activated prob
    ct_template_gesture_likelihood[np.where(ct_template_gesture_chosen_names == true_name)[0][0]] = normal(activated_mu, activated_sigma)

    return ct_template_gesture_likelihood, ct_template_gesture_chosen_names

