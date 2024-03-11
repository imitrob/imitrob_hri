import numpy as np
from imitrob_hri.merging_modalities.configuration import ConfigurationDefault
from imitrob_hri.merging_modalities.utils import *
from copy import deepcopy

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
        template_names (Float[]) OR manage as .names
    '''
    def __init__(self, p=np.array([]), template_names=[], c=None):
        if c == 'default':
            c = ConfigurationDefault()
        assert c is not None
        self.c = c

        # handle if probabilities not given
        if len(p) == 0: self.p = np.zeros(len(template_names))
        
        assert len(p) == len(template_names), f"p {p} != template_names {template_names}"
        
        if len(p) > 0:
            for tn in template_names:
                assert isinstance(tn, str) and tn[0] != '0', f"template name not string: {tn}"
            
            for n,p_ in enumerate(p):
                if isinstance(p_, type(None)):
                    raise Exception("ProbsVector init with {p} and {template_names}")
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

    def discard_two_maxes(self):
        r = max(self.p) - self.p
        if sum(r==0)>1:
            # print(self.p)
            # print(r)
            # print(sum(r==0))
            # print(self.names)
            # input("??")
            return True
        return False

    @property
    def max(self):
        if self.discard_two_maxes(): return None

        if self.empty: return None
        return self.template_names[np.argmax(self.p)]

    @property
    def max_prob(self):
        if self.discard_two_maxes(): return None
        return np.max(self.p)

    @property
    def max_id(self):    
        if self.discard_two_maxes(): return None          
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
            return self.max_id
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
            self.conclusion = 'use'
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
            self.conclusion = 'use'
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

    def add(self, name, p=None):
        """Adds name(s) and its probability(ies) to the vector

        Args:
            name (str OR str[])
            p (float or float[])
        """
        # adds as another probsvector
        if p is None and isinstance(name, ProbsVector):
            p = name.p
            name = name.names

        # add list
        if isinstance(name, (np.ndarray, list, tuple)):
            # nothing to add
            if len(p) == 0: return
            
            names = name
            probs = p
            assert len(names) == len(probs)
            assert isinstance(names[0], str) and isinstance(probs[0], (float, int)), f"Not the right format: {names}. {probs}"
            for n_,p_ in zip(names, probs):
                self.add_single(n_,p_)
        # add single
        elif isinstance(name, str) and isinstance(p, (int, float)):
            self.add_single(name,p) 
        else: raise Exception(f"Not the right format name: {name}, p: {p}")

    def add_single(self, name, p):
        """Adds name ant its probability to the vector

        Args:
            name (str)
            p (float)
        """
        assert p is not None

        list_names = list(self.template_names)
        list_names.append(name)
        self.template_names = list_names
        list_p = list(self.p)
        list_p.append(p)
        self.p_ = np.array(list_p)

    @property
    def empty(self):
        if len(self.names) == 0:
            return True
        else: 
            return False

    def pop(self, id):
        ''' Pops single id '''
        assert isinstance(self.template_names, list)
        assert isinstance(self.p, np.ndarray)
        template = self.template_names.pop(id)
        p = self.p[id]
        self.p = np.delete(self.p, id)
        return template, p
    
    def prob_for_entity(self, name):
        for n,name_ in enumerate(self.names):
            if name == name_:
                return self.p[n]
        raise Exception(f"name {name} not found in {self.names}")

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
        if self.discard_two_maxes(): return None
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


if __name__ == '__main__':
    pv = ProbsVector(p=[0.1,0.2,0.3],template_names=['mark', 'park', 'quark'], c='default')
    print(f"{pv}")