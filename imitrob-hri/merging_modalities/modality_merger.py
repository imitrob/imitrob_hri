
import numpy as np
from collections import deque

from utils import *
import globals as g

"""
class ModalityReceiver():
    ''' Is this already implemented?
    Timeseries classification before merging
    '''
    def __init__(self):

        self.language_prob_timeseries = deque(maxlen=bufferlen)
        self.gesture_prob_timeseries = deque(maxlen=bufferlen)


    def new_language_input(self, i):
        self.language_prob_timeseries.append(i)
    
    def new_gestures_input(self, i):
        self.gesture_prob_timeseries.append(i)

    def hotword_trigger
"""

class ProbsVector():
    ''' Making decisions based on probability vector
    Parameters:
        p (Float[]): Probabilities vector
        - match() - Is there probable item that matches?
        - resolve() - How to resolve, if probs. don't match
        - clear - all clear actions
        - unsure - all unsure actins
        - negative - all not present actions
        - activated - action most probable or None if more clear actions or no clear action
        - 
    '''
    def __init__(self, p=[], action_names=[]):
        # Handle if action names not given
        if len(action_names) == 0: action_names = g.action_names
        # handle if probabilities not given
        if len(p) == 0: self.p = np.zeros(len(action_names))
        
        assert len(p) == len(action_names)

        self.p = np.array(p)
        self.action_names = action_names
        assert isinstance(self.p, np.ndarray) and isinstance(self.p[0], float)
        self.conclusion = None
        assert g.match_threshold
        try:
            g.match_threshold, g.clear_threshold, g.unsure_threshold, g.diffs_threshold
        except NameError:
            raise Exception("Some threshold value not defined: (match_threshold, clear_threshold, unsure_threshold, diffs_threshold)")

    # Probabilities are classified into three categories
    # 1) clear
    @property
    def clear(self):
        return [self.action_names[id] for id in self.clear_id]

    @property
    def clear_id(self):
        return self.get_probs_in_range(g.clear_threshold, 1.01)

    @property
    def clear_probs(self):
        return self.p[self.clear_id]
        
    # 2) unsure
    @property
    def unsure(self):
        return [self.action_names[id] for id in self.unsure_id]

    @property
    def unsure_id(self):
        return self.get_probs_in_range(g.unsure_threshold, g.clear_threshold)

    @property
    def unsure_probs(self):
        return self.p[self.unsure_id]

    # 3) negative
    @property
    def negative(self):
        return [self.action_names[id] for id in self.negative_id]

    @property
    def negative_id(self):
        return self.get_probs_in_range(-0.01, g.unsure_threshold)

    @property
    def negative_probs(self):
        return self.p[self.negative_id]

    #
    def __str__(self):
        return self.info()

    def info(self):
        cls = self.clear
        clsp = self.clear_probs
        s1, s2, s3 = '', '', ''
        for i in range(len(cls)):
            if cls[i] == self.activated:
                s1 += f"{cc.W}{cls[i]}{cc.E} {clsp.round(2)[i]}, "
            else:
                s1 += f"{cls[i]} {clsp.round(2)[i]}, "
        uns = self.unsure
        unsp = self.unsure_probs
        for i in range(len(uns)):
            s2 += f"{uns[i]} {unsp.round(2)[i]}, "        
        neg = self.negative
        negp = self.negative_probs
        for i in range(len(neg)):
            s3 += f"{neg[i]} {negp.round(2)[i]}, "

        return f"{cc.H}Clear: {cc.E} {s1}\n{cc.H}Unsure:{cc.E} {s2}\n{cc.H}Negative: {cc.E}{s3}\n-> {cc.B}{self.conclude()}{cc.E}"

    @property
    def max(self):
        return self.action_names[np.argmax(self.p)]

    @property
    def max_prob(self):
        return np.max(self.p)

    @property
    def max_id(self):    
        return np.argmax(self.p)

    @property
    def diffs(self):
        ''' Difference between max value and others & max value is increased to be discarded by further evaluation
        '''
        maxid = np.argmax(self.p)
        r = max(self.p) - self.p
        r[maxid] += 1.
        return r
    
    def diffs_above_threshold(self):
        return (self.diffs > g.diffs_threshold).all()

    @property
    def activated_id(self):
        ''' Action to be activated:
            1. Information must be clear
            2. Range between other actions must be above threshold
        Returns:
            activated action (String) or None
        '''
        if len(self.clear) == 1 and self.diffs_above_threshold():
            return self.clear_id[0] 

    @property
    def activated(self):
        if self.activated_id is not None: return self.action_names[self.activated_id]

    @property
    def activated_prob(self):
        if len(self.clear) == 1 and self.diffs_above_threshold():
            return self.p[self.clear_id[0]] 
        else: return 0.0
    
    @property
    def single_clear_id(self):
        return self.clear_id[0] if len(self.clear) == 1 else None

    def is_match(self):
        if self.activated != None and self.p[self.single_clear_id] > g.match_threshold:
            return True
        else:
            return False

    def resolve(self):
        ''' Checks and returns a way how to resolve
        Returns:
            name of item (String): Use item, OR
            ['name item 1', 'name item 2'] ([String, String]): Ask choose new item, OR
            None: Don't understand, ask again (NoneType)
        '''
        if self.activated is not None: # Is there single most probable items?
            self.conclusion = 'resolve use'
            return self.activated
        elif self.clear != []: # Is there more most probable items?
            self.conclusion = 'ask choose'
            return self.clear
        else: # No probability above threshold, Ask again
            self.conclusion = 'ask again'
            return None
        

    def match(self):
        ''' Checks and returns if there is an item to match
        Returns:
            name of item (String): Matched, OR
            None (NoneType): Not Matched
        '''
        if self.is_match():
            self.conclusion = 'match use'
            return self.activated
        else:
            self.conclusion = 'not matched - need to resolve'
            return None

    def get_probs_in_range(self, p_min, p_max):
        r = []
        for n,p_ in enumerate(self.p):
            if p_min <= p_ < p_max:
                r.append(n)
        return r      

    @property
    def conclusion_use(self):
        if self.conclusion in ['match use', 'resolve use']:
            return True
        else:
            return False  
        
    def conclude(self):
        if self.is_match():
            self.match()
        else:
            self.resolve()
        return self.conclusion

class MultiProbsVector():
    ''' Saving multiple ProbsVectors '''
    def __init__(self, pv):
        self.pv = pv

    def __str__(self):
        s = ''
        for n,item in enumerate(self.pv):
            s += f"{cc.W}{n}:{cc.E}\n{item}"
        return s
    
    @property
    def conclusion(self):
        conclusions = []
        for n,item in enumerate(self.pv):
            conclusions.append(item.conclude())

        if 'ask again' in conclusions:
            return 'ask again'
        elif 'ask choose' in conclusions:
            return 'ask choose'
        elif 'resolve use' in conclusions:
            return 'resolve use'
        elif 'match use' in conclusions:
            return 'match use'
        else: raise Exception(f"conclusion not in options. conclusions: {conclusions}")

    @property
    def conclusion_use(self):
        if self.conclusion in ['match use', 'resolve use']:
            return True
        else:
            return False  
    
    @property
    def activated(self):
        acts = []
        for n,item in enumerate(self.pv):
            acts.append(item.activated)
        return acts

class SingleTypeModalityMerger():
    def __init__(self, cl_prior = 0.99, cg_prior = 0.8, fun='mul', names=[]):
        self.prior_confidence_language = cl_prior
        self.prior_confidence_gestures = cg_prior
        self.fun = fun
        self.names = names
        assert self.fun in np.array(['mul', 'abs_sub'])

    def merge(self, ml, mg):
        po = self.match(ml, mg)
        po.conclude()
        if po.is_match():
            return po
        else:
            return self.resolve(ml, mg)

    def match(self, cl, cg):
        cl = self.prior_confidence_language * np.array(cl)
        cg = self.prior_confidence_gestures * np.array(cg)    
        
        return self.probs_similarity_magic(cl, cg)

    def resolve(self, cl, cg):
        cl = self.prior_confidence_language * np.array(cl)
        cg = self.prior_confidence_gestures * np.array(cg)    
        
        return self.probs_similarity_magic(cl, cg)

    def probs_similarity_magic(self, cl, cg):
        assert len(cl) == len(cg)
        cm = np.zeros(len(cl))
        for n,(l,g) in enumerate(zip(cl, cg)):
            cm[n] = getattr(self, self.fun)(l, g)
            #getattr(float(l), self.fun)(g)
        return ProbsVector(cm, self.names)
    
    def add_2(self, l, g):
        return abs(l + g)/2
    
    def mul(self, l, g):
        return l * g

class SelectionTypeModalityMerger(SingleTypeModalityMerger):
    def __init__(self, cl_prior = 0.9, cg_prior = 0.99, fun='mul', names=[]):
        super().__init__(cl_prior=cl_prior, cg_prior=cg_prior, fun=fun, names=names)

    def merge(self, ml, mg, aor):
        return MultiProbsVector(self.match(ml, mg, aor))
        po = self.match(ml, mg)
        if po.is_match(): return po
        else: return self.resolve(ml, mg)

    def cross_match(self, cl, cg, aor):
        ''' Alignment function 
        Now linear alignment:
        L: -- x ------ x ------ x ------
        G: --- x ---- x ------ x -------
        Future:
        L: -- x ------ x ------ x ------
        G: ------- x ------ x ----------
        '''
        instances_l, instances_g = len(cl), len(cg)
        if instances_l == instances_g:
            # TODO: Do cross matching, now it is linear match
            r = []
            for i in range(len(cl)):
                r.append(self.probs_similarity_magic(cl[i], cg[i]))
            return r
        else: raise Exception("Not Implemented Error")
        '''
        elif abs(instances_l - instances_g) > 0:
            # where is less instances
            
            argmin_id = np.argmin(np.array([instances_l, instances_g]))
            c_less = [cl, cg][argmin_id]
            c_more = [cl, cg][int(not argmin_id)] 

            c_save = []
            for n,ci in enumerate(c_less):
                # 1-st comparison
                comp_1 = self.probs_similarity_magic(c_more[n], c_less[n])
                # OR 2-nd comparison
                comp_2 = self.probs_similarity_magic(c_more[n+1], c_less[n])

                c_save.append(max(comp_1, comp_2))
            if min(instances_l - instances_g) < aor: # one is missing
                
                return c_save
            elif min(instances_l - instances_g) > aor: # one is over
                
                # need for discard
                nid = np.argmin(c_save[0,-1])
                return c_save.pop(nid)
            else:
                raise Exception("Not valid")
        else:
            raise Exception("NotImplementedErrors")
        '''

    def match(self, cl, cg, ao):
        ''' This fun. replaces the match function from SingleTypeModalityMerger 
            This fun. matches all object from l & g. There can be more observation instances for one action.
        '''
        cl = self.prior_confidence_language * np.array(cl)
        cg = self.prior_confidence_gestures * np.array(cg)
        instances_l = len(cl)
        instances_g = len(cg)
        match_decision_probs = []
        # Matches for different possible range values which can construct template action
        # Now only one number of objects 
        #for ao in aor:
        if True:
            '''
            # one instance is missing entirely
            if ao > instances_l and ao > instances_g:
                # TODO: Or some low probs
                match_decision_probs.append( 0. )
            
            # one (or more) instance modality over missing entirely
            elif ao > instances_l and ao > instances_g:
                how_many_instances_to_discard = ao - instances_l 
                low_probs = self.cross_match(cl, cg, ao)
                low_probs.discard_smallest(how_many_instances_to_discard)
                match_decision_probs.append(low_probs) # match
            
            # n of observed object instances matches 
            elif instances_l == instances_g == ao:
            '''
            if True:
                match_decision_probs.append(self.cross_match(cl, cg, ao))
                
        '''
        sums = []
        print("match_decision_probs", match_decision_probs[0][0], match_decision_probs[0][1])
        print(match_decision_probs[0][0].activated)
        print(match_decision_probs[0][1].activated)
        
        for match_probs in match_decision_probs:    
            sums.append(sum([pp.activated_prob for pp in match_probs]))
        aor_id = np.argmax(np.array(sums))
        '''
        pos = match_decision_probs[0]
        return pos

    def resolve(self, cl, cg):
        ''' This fun. replaces the match function from SingleTypeModalityMerger '''
        cl = self.prior_confidence_language * np.array(cl)
        cg = self.prior_confidence_gestures * np.array(cg)    
        
        return self.probs_similarity_magic(cl, cg)




class ModalityMerger():
    def __init__(self, action_names, object_names):
        self.action = SingleTypeModalityMerger(names=action_names)
        self.selection = SelectionTypeModalityMerger(names=object_names)
        self.distance = SingleTypeModalityMerger()
        self.angular = SingleTypeModalityMerger()

    def get_num_of_object_needed_for_action_magic(self, action):
        return 2

    def get_num_of_distances_needed_for_action_magic(self, action):
        return 0

    def get_num_of_angular_needed_for_action_magic(self, action):
        return 0

    @staticmethod
    def is_zeros(arr, threshold=1e-3):
        return np.allclose(arr, np.zeros(arr.shape), atol=threshold)
    
    @staticmethod
    def is_one_only(arr):
        larr = list(arr)
        lenarr = len(larr)
        if larr.count(1) == 1 and larr.count(0) == lenarr - 1:
            return True
        return False


    def feedforward(self, language_sentence, gesture_sentence, epsilon=0.05, gamma=0.5):
        '''
        
        '''
        # A.) Data preprocessing
        # 1. Add epsilon
        if self.is_zeros(language_sentence.target_action):
            language_sentence.target_action += epsilon
        if self.is_zeros(gesture_sentence.target_action):
            gesture_sentence.target_action += epsilon
        for n,o in enumerate(language_sentence.target_objects):
            if self.is_zeros(o):
                language_sentence.target_objects[n] += epsilon
        for n,o in enumerate(gesture_sentence.target_objects):
            if self.is_zeros(o):
                gesture_sentence.target_objects[n] += epsilon

        # 2. Add gamma, if language includes one value
        if self.is_one_only(language_sentence.target_action):
            language_sentence.target_action += gamma
            language_sentence.target_action = np.clip(language_sentence.target_action, 0, 1)
        for n,o in enumerate(language_sentence.target_objects):
            if self.is_one_only(o):
                language_sentence.target_objects[n] += gamma
                language_sentence.target_objects[n] = np.clip(o, 0, 1)

        print(language_sentence.target_action,language_sentence.target_objects, gesture_sentence.target_action, gesture_sentence.target_objects)

        # B.) Merging
        # 1. Action merge
        action_po = self.action.merge(language_sentence.target_action, gesture_sentence.target_action)
        if not action_po.conclusion_use:
            return action_po.conclusion # ask user to choose or repeat 
        
        aon = self.get_num_of_object_needed_for_action_magic(action_po.activated)
        if aon > 0:
            assert aon == len(language_sentence.target_objects) == len(gesture_sentence.target_objects)
            # 2. Selection merge
            selection_po = self.selection.merge(language_sentence.target_objects, gesture_sentence.deictic_confidence * gesture_sentence.target_objects, aon)
            
            if not selection_po.conclusion_use:
                return selection_po.conclusion # ask user to choose or repeat 
        else:
            selection_po = ProbsVector()

        if self.get_num_of_distances_needed_for_action_magic(action_po.activated):
            # 3. Distance parameters merge
            distance_po = np.average(language_sentence.target_distance, gesture_sentence.target_distance) 
        else:
            distance_po = []

        if self.get_num_of_angular_needed_for_action_magic(action_po.activated):
            # 4. Angular parameters merge
            angular_po = np.average(language_sentence.target_action, gesture_sentence.target_action) 
        else:
            angular_po = []

        return f'Action: {action_po.activated}, What to do: {action_po.conclusion} \n Objects:{selection_po.activated}, What to do: {selection_po.conclusion}'
        
class UnifiedSentence():
    def __init__(self, target_action, target_objects=[], distance_params=[], angular_params=[], deictic_confidence=1.0):
        self.target_action = np.array(target_action)
        self.target_objects = np.array(target_objects)
        self.distance_params = np.array(distance_params)
        self.angular_params = np.array(angular_params)

        self.deictic_confidence = deictic_confidence

    #def 

