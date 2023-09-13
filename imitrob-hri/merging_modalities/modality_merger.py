
import numpy as np
from collections import deque

try:
    from utils import *
except ModuleNotFoundError:
    from merging_modalities.utils import *

import os
from os import listdir
from os.path import isfile, join

import sys; sys.path.append("..")
from nlp_new.nlp_utils import make_conjunction, to_default_name, create_template
from copy import deepcopy

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
        - clear - all clear templates
        - unsure - all unsure actins
        - negative - all not present templates
        - activated - template most probable or None if more clear templates or no clear template
        - 
    '''
    def __init__(self, p=np.array([]), template_names=[], c=None):
        assert c is not None
        self.c = c

        # handle if probabilities not given
        if len(p) == 0: self.p = np.zeros(len(template_names))
        
        assert len(p) == len(template_names), f"p {p} != template_names {template_names}"
        
        if len(p) > 0:
            for tn in template_names:
                assert isinstance(tn, str) and tn[0] != '0', f"template name not string: {tn}"

        self.p = np.array(p)
        self.template_names = template_names
        assert isinstance(self.p, np.ndarray) and (len(self.p) == 0 or isinstance(self.p[0], float))
        self.conclusion = None
        assert self.c.match_threshold
        try:
            self.c.match_threshold, self.c.clear_threshold, self.c.unsure_threshold, self.c.diffs_threshold
        except NameError:
            raise Exception("Some threshold value not defined: (match_threshold, clear_threshold, unsure_threshold, diffs_threshold)")

    # Probabilities are classified into three categories
    # 1) clear
    @property
    def clear(self):
        return [self.template_names[id] for id in self.clear_id]

    @property
    def clear_id(self):
        return self.get_probs_in_range(self.c.clear_threshold, 1.01)

    @property
    def clear_probs(self):
        return self.p[self.clear_id]
        
    # 2) unsure
    @property
    def unsure(self):
        return [self.template_names[id] for id in self.unsure_id]

    @property
    def unsure_id(self):
        return self.get_probs_in_range(self.c.unsure_threshold, self.c.clear_threshold)

    @property
    def unsure_probs(self):
        return self.p[self.unsure_id]

    # 3) negative
    @property
    def negative(self):
        return [self.template_names[id] for id in self.negative_id]

    @property
    def negative_id(self):
        return self.get_probs_in_range(-0.01, self.c.unsure_threshold)

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
        return self.template_names[np.argmax(self.p)]

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
        return (self.diffs > self.c.diffs_threshold).all()

    @property
    def activated_id(self):
        ''' template to be activated:
            1. Information must be clear
            2. Range between other templates must be above threshold
        Returns:
            activated template (String) or None
        '''
        if len(self.clear) == 1 and self.diffs_above_threshold():
            return self.clear_id[0] 

    @property
    def activated(self):
        if self.activated_id is not None: return self.template_names[self.activated_id]

    @property
    def activated_prob(self):
        if len(self.clear) == 1 and self.diffs_above_threshold():
            return self.p[self.clear_id[0]] 
        else: return 0.0
    
    @property
    def single_clear_id(self):
        return self.clear_id[0] if len(self.clear) == 1 else None

    def is_match(self):
        if self.activated != None and self.p[self.single_clear_id] > self.c.match_threshold:
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
    
    @property
    def names(self):
        return self.template_names
    
    @names.setter
    def names(self, n):
        self.template_names = n

    @property
    def p(self):
        return self.p_

    @p.setter
    def p(self, p_):
        p_ = np.array(p_)
        assert p_.ndim == 1
        self.p_ = p_

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
    def __init__(self, cl_prior = 0.99, cg_prior = 0.8, fun='mul', names=[], c=None):
        self.prior_confidence_language = cl_prior
        self.prior_confidence_gestures = cg_prior
        self.fun = fun
        self.names = names
        assert self.fun in np.array(['mul', 'abs_sub'])

        assert c is not None
        self.c = c

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
        return ProbsVector(cm, self.names, self.c)
    
    def add_2(self, l, g):
        return abs(l + g)/2
    
    def mul(self, l, g):
        return l * g

"""
class SelectionTypeModalityMerger(SingleTypeModalityMerger):
    def __init__(self, cl_prior = 0.9, cg_prior = 0.99, fun='mul', names=[]):
        super().__init__(cl_prior=cl_prior, cg_prior=cg_prior, fun=fun, names=names)

    def merge(self, ml, mg):
        return MultiProbsVector(self.match(ml, mg))
        po = self.match(ml, mg)
        if po.is_match(): return po
        else: return self.resolve(ml, mg)

    def cross_match(self, cl, cg):
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

    def match(self, cl, cg):
        ''' This fun. replaces the match function from SingleTypeModalityMerger 
            This fun. matches all selection from l & g. There can be more observation instances for one template.
        '''
        cl = self.prior_confidence_language * np.array(cl)
        cg = self.prior_confidence_gestures * np.array(cg)
        instances_l = len(cl)
        instances_g = len(cg)
        match_decision_probs = []
        # Matches for different possible range values which can construct template template
        # Now only one number of selections 
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
            
            # n of observed selection instances matches 
            elif instances_l == instances_g == ao:
            '''
            if True:
                match_decision_probs.append(self.cross_match(cl, cg))
                
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
"""



class ModalityMerger():
    def __init__(self, c):
        assert len(c.ct_names.keys()) > 0
        self.c = c
        self.mms = {}
        for ct in c.ct_names.keys():
            mm = SingleTypeModalityMerger(names=c.ct_names[ct], c=self.c)
            self.mms[ct] = mm
        
        self.compare_types = c.ct_names.keys()

    def get_cts_type_objs(self):
        cts = []
        for compare_type in self.compare_types:
            cts.append(getattr(self, compare_type))
        return cts

    def __str__(self):
        s = ''
        for mmn, mm in zip(self.compare_types, self.get_cts_type_objs()):
            s += mmn.capitalize() + ': ' + str(mm.names) + '\n'
        return f"** Modality merge summary: **\n{s}**"

    def get_all_templates(self):
        return self.c.ct_names['template']
    '''
    def get_names_for_compare_type(self, compare_type):
        return {
            '-': ['abc', 'def', 'ghi']
        }[compare_type]

    def get_num_of_selection_needed_for_template_magic(self, template):
        return 2

    def get_num_of_distances_needed_for_template_magic(self, template):
        return 0

    def get_num_of_angular_needed_for_template_magic(self, template):
        return 0
    '''
    @staticmethod
    def is_zeros(arr, threshold=1e-3):
        return np.allclose(arr, np.zeros(arr.shape), atol=threshold)
    
    @staticmethod
    def is_one_only(arr):
        arr = np.array(arr)
        assert len(arr.shape) == 1, f"Only 1D arrays possible, shape: {arr.shape}"
        larr = list(arr)
        lenarr = len(larr)
        if larr.count(1) == 1 and larr.count(0) == lenarr - 1:
            return True
        return False

    def preprocessing(self, ls, gs, epsilon, gamma):
        ''' Data preprocessing '''
        # 1. Add epsilon

        for ct in self.c.ct_names.keys():
            if self.is_zeros(ls[ct].p):
                ls[ct].p += epsilon
            if self.is_zeros(gs[ct].p):
                gs[ct].p += epsilon

        # 2. Add gamma, if language includes one value
        for ct in self.c.ct_names.keys():
            if self.is_one_only(ls[ct].p):
                ls[ct].p += gamma
                ls[ct].p = np.clip(ls[ct].p, 0, 1)

        return ls, gs

    def feedforward(self, language_sentence, gesture_sentence, epsilon=0.05, gamma=0.5):
        '''
        
        '''
        # A.) Data preprocessing
        language_sentence, gesture_sentence = self.preprocessing(language_sentence, gesture_sentence, epsilon, gamma)

        # B.) Merging
        # 1. template merge
        template_po = self.template.merge(language_sentence.target_template, gesture_sentence.target_template)
        if not template_po.conclusion_use:
            return template_po.conclusion # ask user to choose or repeat 
        
        # Note: aon deleted
        aon = self.get_num_of_selection_needed_for_template_magic(template_po.activated)
        if aon > 0:
            assert aon == len(language_sentence.target_selections) == len(gesture_sentence.target_selections), f"Target selections num don't match: lan {language_sentence.target_selection}, ges {gesture_sentence.target_selection}"
            # 2. Selections merge
            selection_po = self.selection.merge(language_sentence.target_selections, gesture_sentence.deictic_confidence * gesture_sentence.target_selection)
            
            if not selection_po.conclusion_use:
                return selection_po.conclusion # ask user to choose or repeat 
        else:
            selection_po = ProbsVector()

        if self.get_num_of_distances_needed_for_template_magic(template_po.activated):
            # 3. Distance (compare type) parameters merge
            distance_po = np.average(language_sentence.target_distance, gesture_sentence.target_distance) 
        else:
            distance_po = []

        if self.get_num_of_angular_needed_for_template_magic(template_po.activated):
            # 4. Angular (compare type) parameters merge
            angular_po = np.average(language_sentence.target_template, gesture_sentence.target_template) 
        else:
            angular_po = []

        return f'template: {template_po.activated}, What to do: {template_po.conclusion} \n selections:{selection_po.activated}, What to do: {selection_po.conclusion}'
    
    def single_modality_merge(self, compare_type, lsp, gsp):
        # Get single modality merger
        if compare_type in self.compare_types:
            mm = self.mms[compare_type] 
        else: raise Exception("compare_type not in self.compare_types")
        return mm.merge(lsp, gsp)
    
    def is_ct_visible(self, s_, ct_target, threshold = 0.1):
        for ct in self.compare_types:
            if sum(s_[ct].p) > threshold:
                if ct == ct_target:
                    return True

        return False
    
    def feedforward2(self, ls, gs, scene, epsilon=0.05, gamma=0.5, alpha_penal=0.9, model=1):
        ''' 
            gs: gesture_sentence, ls: language_sentence

            templates: point, pick, place
            compare_types: selections, storages, distances, ...

            alpha - penalizes if template does/doesn't have compare type which is in the sentence
        '''
        DEBUGdata = []
        # A.) Data preprocessing
        ls, gs = self.preprocessing(ls, gs, epsilon, gamma)

        # B.) Merging
        # 1. Compare types independently
        S_naive = {}
        for compare_type in self.compare_types: # storages, distances, ...
            # single compare-type merger e.g. [box1, cube1, ...] (probs.)
            S_naive[compare_type] = self.single_modality_merge(compare_type, \
                                        ls[compare_type].p,
                                        gs[compare_type].p)
        # 2. Penalize likelihood for every template
        templates = self.get_all_templates()
        template_ct_penalized = deepcopy(S_naive['template']) # 1D (templates)
        template_ct_penalized_real = deepcopy(S_naive['template']) # 1D (templates)
        
        if self.c.DEBUG:
            print(f"Template BEFORE: {template_ct_penalized.p}")

        if model > 1:

            for nt, template in enumerate(templates): # point, pick, place
                template_obj = create_template(template)

                alpha = 1.0
                beta = 1.0
                beta_real = 1.0
                for nct, compare_type in enumerate(self.compare_types): # selections, storages, distances, ...
                    # if compare type is missing in sentence or in template -> penalize
                    compare_type_in_sentence = (self.is_ct_visible(ls, compare_type) or self.is_ct_visible(gs, compare_type))
                    compare_type_in_template = template_obj.has_compare_type(compare_type) 

                    #print("template", template, "compare_type", compare_type, "compare_type_in_template", compare_type_in_template, "compare_type_in_sentence", compare_type_in_sentence)
                    #input()
                    if compare_type_in_template != compare_type_in_sentence:
                        alpha *= alpha_penal
                        

                if model > 2:
                    beta = 0.0
                    for o in scene.selections:
                        for s in scene.storages:
                            if template_obj.is_feasible(o, s):
                                beta = 1.0
                    # feasible template on current scene?
                    '''
                    if model > 2: 
                        #print("template_obj.has_compare_type(compare_type):", template_obj.name, compare_type, template_obj.has_compare_type(compare_type))
                        if template_obj.has_compare_type(compare_type) and \
                            compare_type == 'selections': # TEMP FOR DEBUG
                            #                       properties              x         names
                            b = np.ones((len(self.c.ct_properties[compare_type]), len(S_naive[compare_type].names)))

                            
                            b_real = np.ones((len(self.c.ct_properties[compare_type]), len(S_naive[compare_type].names)))
                            if len(self.c.ct_properties[compare_type]) > 0:
                                
                                for n,property_name in enumerate(self.c.ct_properties[compare_type]):
                                    
                                    # check properties, penalize non-compatible ones
                                    b[n], b_real[n] = penalize_properties(template_obj, property_name, compare_type, S_naive[compare_type], self.c, scene)

                                if max(b.prod(1)) != 1.0:
                                    print(f"beta: {max(b.prod(1))}. {compare_type}, t {template_obj.name}")


                                #print("jakoby b  ", b)
                                # BIG QUESTION HOW TO PENALIZE?
                                beta *= max(b.prod(1)) # this is draft how it should look like
                                # look at notes in notebook for more info
                                beta_real *= max(b_real.prod(1)) # this is draft how it should look like
                                if beta != 1:
                                    beta_counter += 1

                            DEBUGdata.append((template, compare_type, self.c.ct_properties[compare_type], S_naive[compare_type].names, b, beta))
                    '''
                template_ct_penalized.p[nt] *= alpha
                template_ct_penalized.p[nt] *= beta

                template_ct_penalized_real.p[nt] *= beta_real
            
        if self.c.DEBUG: 
            print(f"Template AFTER: {template_ct_penalized.p}")
        # hack
        template_ct_penalized.p = np.clip(1.4 * template_ct_penalized.p, 0, 1)
        S_naive['template'] = template_ct_penalized
        
        S_naive['storages'].p = np.clip(1.4 * S_naive['storages'].p, 0, 1)
        
        return S_naive, DEBUGdata
    
def penalize_properties(template_obj, property_name, compare_type, S_naive_c, c, scene):
    '''
    template (String): e.g. 'PickTask'
    property_name (String): e.g. 'reachable', 'pickable', ...
    compare_type (String): e.g. 'template', 'selections', ...
    S_naive_c: Sentence naive merged - single compare type
    '''
    ret = np.ones((len(S_naive_c.p)))
    ret_real = np.ones((len(S_naive_c.p)))
    #print(f"[penalize properties] len items p: {len(S_naive_c.p)}")
    if compare_type == 'selections':
        task_property_penalization = template_obj.task_property_penalization_selections(property_name)
    elif compare_type == 'storages':
        task_property_penalization = template_obj.task_property_penalization_storages(property_name)
    else: raise Exception()

    #if not (len(S_naive_c.p) == 0):
    n = 0
    for p, name in zip(S_naive_c.p, S_naive_c.names):
        #if scene.get_object(name).properties[property_name] is not None:
        penalization, eval = scene.get_object(name).properties[property_name]()
        if task_property_penalization < 1.0:
            
            ret[n] = int(eval) #to see that it works,,,, p * penalization 
            ret_real[n] = p * bool(eval)

        n += 1
    
    return ret, ret_real
    

class MMSentence():
    def __init__(self, L, G, M=None):
        self.L = L # Language
        self.G = G # Gestures
        self.M = M # Merged

    def make_conjunction(self, c):
        '''
        Parameters:
            c (Configuration()) (pointer)
        '''
        for ct in c.ct_names.keys():
            c.ct_names[ct], self.L[ct].p, self.G[ct].p = make_conjunction( \
                                        self.G[ct].names, self.L[ct].names, \
                                        self.G[ct].p, self.L[ct].p, ct=ct)
            self.G[ct].names = c.ct_names[ct]
            self.L[ct].names = c.ct_names[ct]
        # special case: extend to all loaded templates
        for template in ['pick', 'point', 'PutTask']:
            if to_default_name(template) not in c.ct_names['template']:
                c.ct_names['template'] = np.append(c.ct_names['template'], to_default_name(template))
                self.G['template'].p = np.append(self.G['template'].p, 0.0)
                self.L['template'].p = np.append(self.L['template'].p, 0.0)
                self.G['template'].names = c.ct_names['template']
                self.L['template'].names = c.ct_names['template']

    def check_merged(self, y, c, printer=True):
        success = True
        each_ct = []
        for ct in c.ct_names.keys():
            if y[ct] == self.M[ct].activated:
                if printer:
                    print(f"{cc.H}{y[ct]} == {self.M[ct].activated}{cc.E}", end="; ")
            else:
                if printer:
                    print(f"{cc.F}{y[ct]} != {self.M[ct].activated}{cc.E}", end="; ")
            
            if ct in y.keys():
                if y[ct] != self.M[ct].activated:
                    success = False
            each_ct.append(success)
        if printer:
            print()
        return success

    def __str__(self):
        return f"L:\n{self.L['template']}\n{self.L['selections']}\n{self.L['storages']}, G:\n{self.G['template']}\n{self.G['selections']}\n{self.G['storages']}"
