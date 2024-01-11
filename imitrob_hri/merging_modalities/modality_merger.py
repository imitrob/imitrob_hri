
import numpy as np
from collections import deque

try:
    from imitrob_hri.merging_modalities.utils import *
except ModuleNotFoundError:
    from merging_modalities.utils import *

import os
from os import listdir
from os.path import isfile, join

import sys; sys.path.append("..")
from imitrob_hri.imitrob_nlp.nlp_utils import make_conjunction, to_default_name
from imitrob_templates.small_template_factory import create_template
from copy import deepcopy

from teleop_msgs.msg import HRICommand


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
            
            for n,p_ in enumerate(p):
                if type(p_) == str:
                    p[n] = float(p_)    

        self.p = np.array(p)
        self.template_names = template_names
        
        self.p = np.array(self.p, dtype=float) # handle when input self.p is e.g., np.array(["1.0","0.5",...])
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
        if len(self.clear) > 0 and self.diffs_above_threshold():
            return self.clear_id[0]
        #elif (len(self.unsure) > 0 and self.diffs_above_threshold()):
        #    return self.unsure_id[0] 

    @property
    def activated(self):
        if self.activated_id is not None: return self.template_names[self.activated_id]

    @property
    def activated_prob(self):
        if len(self.clear) > 0 and self.diffs_above_threshold():
            return self.p[self.clear_id[0]] 
        else: return 0.0
    
    @property
    def single_clear_id(self):
        return self.clear_id[0] if len(self.clear) == 1 else None

    def is_match(self):
        if self.single_clear_id != None and self.p[self.single_clear_id] > self.c.match_threshold:
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




class EntropyProbsVector(ProbsVector):
    # UNIFORM_ENTROPY_TH = 0.85
    UNIFORM_ENTROPY_TH = 0
    UNIFORM_ENTROPY_TH_LEN_1 = 1.1
    NOISE_TH = 0.05

    def __init__(self, p, template_names=[], c=None):
        super().__init__(p, template_names, c)
        self.recompute_ids()

    def recompute_ids(self):
        clear_th = self.UNIFORM_ENTROPY_TH or normalized_entropy(self.p_)  # fixed threshold or entropy
        if len(self.p_) == 1: clear_th = self.UNIFORM_ENTROPY_TH_LEN_1 # exception for len=1

        dcross_ent = np.asarray(diagonal_cross_entropy(self.p_))
        clear_ids = np.where(dcross_ent < clear_th)[0].tolist()
        unsure_ids = np.where(np.logical_and(dcross_ent >= clear_th, np.asarray(self.p_) > self.NOISE_TH))[0].tolist()

        self._set_ids(clear_ids, unsure_ids)

    def _set_ids(self, clear_ids, unsure_ids):
        self._clear_ids = deepcopy(clear_ids)
        self._unsure_ids = deepcopy(unsure_ids)
        self._negative_ids = deepcopy([i for i in range(len(self.p)) if i not in self._clear_ids + self._unsure_ids])

    @property
    def clear_id(self):
        return self._clear_ids

    @property
    def unsure_id(self):
        return self._unsure_ids

    @property
    def negative_id(self):
        return self._negative_ids

    @property
    def p(self):
        return self.p_

    @p.setter
    def p(self, p_):
        p_ = np.array(p_)
        assert p_.ndim == 1
        self.p_ = p_
        self.recompute_ids()

# from configuration import Configuration1
# epv = EntropyProbsVector([1.], template_names=['a'], c=Configuration1())
# print(f"epv: {epv}")

class NaiveProbsVector(ProbsVector):
    def __init__(self, p, template_names=[], c=None):
        super().__init__(p, template_names, c)   

    @property
    def clear_id(self):
        if len(self.p) == 0: return []
        return [np.argmax(self.p)]

    @property
    def unsure_id(self):
        return []

    @property
    def negative_id(self):
        arange = list(range(len(self.p)))
        if self.clear_id != []:
            arange.remove(self.clear_id[0])        
        return arange

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
        assert self.fun in np.array(['mul', 'add_2', 'entropy', 'baseline', 'entropy_add_2'])

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

    def _get_single_probs_vector(self, p, names):
        return ProbsVector(p, names, self.c)

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
    def __init__(self, c, use_magic):
        assert len(c.mm_pars_names_dict.keys()) > 0
        self.c = c
        self.mms = {}
        for ct in c.mm_pars_names_dict.keys():
            mm = SingleTypeModalityMerger(names=c.mm_pars_names_dict[ct], c=self.c, fun=use_magic)
            self.mms[ct] = mm
        
        self.mm_pars_compulsary = c.mm_pars_names_dict.keys()
        self.use_magic = use_magic

    def get_cts_type_objs(self):
        cts = []
        for compare_type in self.mm_pars_compulsary:
            cts.append(getattr(self, compare_type))
        return cts

    def __str__(self):
        s = ''
        for mmn, mm in zip(self.mm_pars_compulsary, self.get_cts_type_objs()):
            s += mmn.capitalize() + ': ' + str(mm.names) + '\n'
        return f"** Modality merge summary: **\n{s}**"

    def get_all_templates(self):
        return self.c.mm_pars_names_dict['template']
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
        # for ct in self.c.mm_pars_names_dict.keys():
        #     if self.is_zeros(ls[ct].p):
        #         ls[ct].p += epsilon
        #     if self.is_zeros(gs[ct].p):
        #         gs[ct].p += epsilon

        # 2. Add gamma, if language includes one value
        for ct in self.c.mm_pars_names_dict.keys():
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
    
    def naive_modality_merge(self, compare_type, lsp, gsp):
        if compare_type in self.mm_pars_compulsary:
            mm = self.mms[compare_type]
        else:
            raise Exception("compare_type not in self.mm_pars_compulsary")

        merged_p = []
        for lp, gp in zip(lsp, gsp):
            merged_p.append(max(lp, gp))

        return NaiveProbsVector(merged_p, mm.names, mm.c)

    def single_modality_merge(self, compare_type, lsp, gsp):
        # Get single modality merger
        if compare_type in self.mm_pars_compulsary:
            mm = self.mms[compare_type] 
        else: raise Exception("compare_type not in self.mm_pars_compulsary")
        return mm.merge(lsp, gsp)

    def entropy_modality_merge(self, compare_type, lsp, gsp):
        if compare_type in self.mm_pars_compulsary:
            mm = self.mms[compare_type]
        else:
            raise Exception("compare_type not in self.mm_pars_compulsary")

        # penalize according to entropy?
        PENALIZE_BY_ENTROPY = True
        DISCARD_ENTROPY_THRESHOLD = 0.85

        if normalized_entropy(lsp) > DISCARD_ENTROPY_THRESHOLD:
            # lsp = np.ones_like(lsp) * np.finfo(lsp.dtype).eps
            msp = gsp
        elif normalized_entropy(gsp) > DISCARD_ENTROPY_THRESHOLD:
            # gsp = np.ones_like(gsp) * np.finfo(gsp.dtype).eps
            msp = lsp
        else:
            if PENALIZE_BY_ENTROPY:
                # exception for len(1)
                if len(lsp) == 1:
                    penalization_by_entropy_l = 1.
                else:
                    penalization_by_entropy_l = diagonal_cross_entropy(lsp)
                if len(gsp) == 1:
                    penalization_by_entropy_g = 1.
                else:
                    penalization_by_entropy_g = diagonal_cross_entropy(gsp)
                    
                lsp /= penalization_by_entropy_l
                gsp /= penalization_by_entropy_g

            if self.use_magic == 'entropy':
                msp = lsp * gsp  # "merge"
            elif self.use_magic == 'entropy_add_2':
                msp = lsp + gsp
            else:
                raise Exception("TODO?")

        msp /= np.sum(msp)  # normalize

        epv = EntropyProbsVector(msp, mm.names, mm.c)
        return epv 

    def is_ct_visible(self, s_, ct_target, threshold = 0.1):
        for ct in self.mm_pars_compulsary:
            if sum(s_[ct].p) > threshold:
                if ct == ct_target:
                    return True

        return False
    
    def feedforward2(self, ls, gs, scene, epsilon=0.05, gamma=0.5, alpha_penal=0.9, model=1):
        ''' v2 - more general version (testing in process)
            gs: gesture_sentence, ls: language_sentence
            alpha - penalizes if template does/doesn't have compare type which is in the sentence
        '''
        DEBUGdata = []
        # A.) Data preprocessing
        ls, gs = self.preprocessing(ls, gs, epsilon, gamma)

        # B.) Merging
        # 1. Compare types independently
        S_naive = {}
        for compare_type in self.mm_pars_compulsary: # storages, distances, ...
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

            for nt, template in enumerate(templates): # e.g. pick, push, put-into
                template_obj = create_template(template)
                assert template_obj is not None, f"Template {template} not visible!"

                alpha = 1.0
                beta = 1.0
                beta_real = 1.0
                for nct, compare_type in enumerate(self.mm_pars_compulsary): # selections, storages, distances, ...
                    # if compare type is missing in sentence or in template -> penalize
                    compare_type_in_sentence = (self.is_ct_visible(ls, compare_type) or self.is_ct_visible(gs, compare_type))
                    compare_type_in_template = template_obj.has_compare_type(compare_type) 

                    if compare_type_in_template != compare_type_in_sentence:
                        alpha *= alpha_penal
                        
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
                    
                template_ct_penalized.p[nt] *= alpha
                template_ct_penalized.p[nt] *= beta

                template_ct_penalized_real.p[nt] *= beta_real
            
        if self.c.DEBUG: 
            print(f"Template AFTER: {template_ct_penalized.p}")

        template_ct_penalized.recompute_ids()

        # hack
        template_ct_penalized.p = np.clip(1.4 * template_ct_penalized.p, 0, 1)
        S_naive['template'] = template_ct_penalized
        
        S_naive['storages'].p = np.clip(1.4 * S_naive['storages'].p, 0, 1)
        
        return S_naive, DEBUGdata
    
    def feedforward3(self, ls, gs, scene, epsilon=0.05, gamma=0.5, alpha_penal=0.9, model=1, use_magic='entropy'):
        ''' v3 (final version)
            - entropy modality merge as default
            - properties modeled as if_feasible function for template

            gs: gesture_sentence, ls: language_sentence
            alpha - penalizes if template does/doesn't have compare type which is in the sentence
        '''
        DEBUGdata = []
        # A.) Data preprocessing
        ls, gs = self.preprocessing(ls, gs, epsilon, gamma)


        # B.) Merging
        # 1. Compare types independently
        S_naive = {}
        for compare_type in self.mm_pars_compulsary: # storages, distances, ...
            
            # single information only?
            if self.is_zeros(ls[compare_type].p):
                S_naive[compare_type] = deepcopy(gs[compare_type])
            elif self.is_zeros(gs[compare_type].p):
                S_naive[compare_type] = deepcopy(ls[compare_type])

            if use_magic == 'baseline':
                S_naive[compare_type] = self.naive_modality_merge(compare_type, \
                            deepcopy(ls[compare_type].p),
                            deepcopy(gs[compare_type].p) )
            elif use_magic == 'entropy' or use_magic == 'entropy_add_2':
                S_naive[compare_type] = self.entropy_modality_merge(compare_type, \
                                    deepcopy(ls[compare_type].p),
                                    deepcopy(gs[compare_type].p))
            elif use_magic == 'mul' or use_magic == 'add_2':
                S_naive[compare_type] = self.single_modality_merge(compare_type, \
                                    deepcopy(ls[compare_type].p),
                                    deepcopy(gs[compare_type].p))                
            else: raise Exception("Wrong")
            
        # print(f"{cc.H}============================={cc.E}")
        # print(f"{cc.H}==== AFTER Compare types independently ========{cc.E}")
        # print(f"{cc.H}============================={cc.E}")
        # print(f"{ls['template']}\n{ls['selections']}")
        # print(f"{cc.H}============================={cc.E}")
        # print(f"{gs['template']}\n{gs['selections']}")
        # print(f"{cc.H}============================={cc.E}")

        
        # 2. Penalize likelihood for every template
        templates = self.get_all_templates()
        # print("final templates: ", templates)
        template_ct_penalized = deepcopy(S_naive['template']) # 1D (templates)
        template_ct_penalized_real = deepcopy(S_naive['template']) # 1D (templates)
        
        if model > 1:
            for nt, template in enumerate(templates): # e.g. pick, push, put=into
                template_obj = create_template(template)
                assert template_obj is not None, f"Template {template} not visible!"

                alpha = 1.0
                beta = 1.0
                beta_real = 1.0
                for nct, compare_type in enumerate(self.mm_pars_compulsary): # selections, storages, distances, ...
                    # if compare type is missing in sentence or in template -> penalize
                    compare_type_in_sentence = (self.is_ct_visible(ls, compare_type) or self.is_ct_visible(gs, compare_type))
                    compare_type_in_template = template_obj.has_compare_type(compare_type) 

                    if compare_type_in_template != compare_type_in_sentence:
                        alpha *= alpha_penal

                if model > 2:
                    if template_obj.mm_pars_compulsary == ['template', 'selections', 'storages']:
                        beta = 0.0
                        for o in scene.selections:
                            for s in scene.storages:
                                if template_obj.is_feasible(o, s):
                                    beta = 1.0
                    elif template_obj.mm_pars_compulsary == ['template', 'selections']:
                        beta = 0.0
                        for o in scene.selections:
                            if template_obj.is_feasible(o):
                                beta = 1.0
                    elif template_obj.mm_pars_compulsary == ['template']:
                        beta = 1.0
                    else: raise Exception(f"TODO {template_obj.mm_pars_compulsary}")

                template_ct_penalized.p[nt] *= alpha
                template_ct_penalized.p[nt] *= beta

                template_ct_penalized_real.p[nt] *= beta_real
            
        
        if use_magic == 'entropy' or use_magic == 'entropy_add_2':
            template_ct_penalized.recompute_ids()
        # hack - only if mul
        if use_magic == 'mul' or use_magic == 'entropy':
            template_ct_penalized.p = np.clip(1.4 * template_ct_penalized.p, 0, 1)
            S_naive['storages'].p = np.clip(1.4 * S_naive['storages'].p, 0, 1)
        S_naive['template'] = template_ct_penalized
        
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
        task_property_penalization = template_obj.task_property_penalization_target_objects(property_name)
    elif compare_type == 'storages':
        task_property_penalization = template_obj.task_property_penalization_target_storages(property_name)
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
        for ct in c.mm_pars_names_dict.keys():
            c.mm_pars_names_dict[ct], self.L[ct].p, self.G[ct].p = make_conjunction( \
                                        self.G[ct].names, self.L[ct].names, \
                                        self.G[ct].p, self.L[ct].p, ct=ct,
                                        keep_only_items_in_c_templates=True,
                                        c_templates=c.mm_pars_names_dict[ct])
            self.G[ct].names = c.mm_pars_names_dict[ct]
            self.L[ct].names = c.mm_pars_names_dict[ct]
        # special case: extend to all loaded templates (from files)
        # for template in ['pick', 'point', 'PutTask']:
        #     if to_default_name(template) not in c.mm_pars_names_dict['template']:
        #         c.mm_pars_names_dict['template'] = np.append(c.mm_pars_names_dict['template'], to_default_name(template))
        #         self.G['template'].p = np.append(self.G['template'].p, 0.0)
        #         self.L['template'].p = np.append(self.L['template'].p, 0.0)
        #         self.G['template'].names = c.mm_pars_names_dict['template']
        #         self.L['template'].names = c.mm_pars_names_dict['template']

    def check_merged(self, y, c, printer=True):
        success = True
        for ct in c.mm_pars_names_dict.keys():
            if y[ct] == self.M[ct].activated:
                if printer:
                    print(f"{cc.H}{y[ct]} == {self.M[ct].activated}{cc.E}", end="; ")
            else:
                if printer:
                    print(f"{cc.F}{y[ct]} != {self.M[ct].activated}{cc.E}", end="; ")
            
            if ct in y.keys():
                if y[ct] != self.M[ct].activated:
                    success = False
        if printer: print()
        return success
    
    def get_true_and_pred(self, y, c, max_only=False):
        '''
        
        '''
        y_true_cts, y_pred_cts = [], []
        for ct in c.mm_pars_names_dict.keys():
            if max_only:
                y_true_cts.append(str(y[ct]))
                y_pred_cts.append(str(self.M[ct].max))
            else:
                y_true_cts.append(str(y[ct]))
                y_pred_cts.append(str(self.M[ct].activated))
        return y_true_cts, y_pred_cts



    def __str__(self):
        return f"L:\n{self.L['template']}\n{self.L['selections']}\n{self.L['storages']}, G:\n{self.G['template']}\n{self.G['selections']}\n{self.G['storages']}"

    def merged_part_to_HRICommand(self):
        return self.to_HRICommand(self.M)
    
    @staticmethod
    def to_HRICommand(M):
        ''' From M generates HRICommand
            There is a better way how to do this
            1. create dict
            2. change keys
            3. assign values from M
        '''
        
        # d =  {"target_action": M["template"].activated, "target_object": M["selections"].activated, "target_storage": M["selections"].activated, 
        #       "actions": ["pick", "release", "pass", "point"], "action_probs": ["1.0", "0.05", "0.1", "0.15"], "action_timestamp": 0.0, "objects": ["cube_holes_od_0", "wheel", "sphere"], "object_probs": [1.0, 0.1, 0.15], "object_classes": ["object"], "parameters": ""}'
        
        s  = '{'
        s += f'"target_action": "{M["template"].activated}", '
        s += f'"target_object": "{M["selections"].activated}", '
        s += f'"target_storage": "{M["storages"].activated}", '
        
        s_1 = []
        for nm in M["template"].names:
            s_1.append(f'"{nm}"')
        s_1s = ', '.join(s_1) 
        if len(s_1s) > 0: s += f'"actions": [{s_1s}], '
        s_2 = []
        for nm in M["template"].p:
            s_2.append(f'"{nm}"')
        s_2s = ', '.join(s_2) 
        if len(s_2s) > 0: s += f'"action_probs": [{s_2s}], '

        s_3 = []
        for nm in M["selections"].names:
            s_3.append(f'"{nm}"')
        s_3s = ', '.join(s_3) 
        if len(s_3s) > 0: s += f'"objects": [{s_3s}], '
        s_4 = []
        for nm in M["selections"].p:
            s_4.append(f'"{nm}"')
        s_4s = ', '.join(s_4) 
        if len(s_4s) > 0: s += f'"object_probs": [{s_4s}], '

        s_5 = []
        for nm in M["storages"].names:
            s_5.append(f'"{nm}"')
        s_5s = ', '.join(s_5) 
        
        if len(s_5s) > 0: s += f'"storages": [{s_5s}], '
        s_6 = []
        for nm in M["storages"].p:
            s_6.append(f'"{nm}"')
        s_6s = ', '.join(s_6) 
        if len(s_6s) > 0: s += f'"storage_probs": {s_6s},'

        s += ' "parameters": "" }'
        
        return HRICommand(data=[s])