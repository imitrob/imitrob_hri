#!/usr/bin/env python
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner
"""
import logging
from typing import Any

import numpy as np

from imitrob_hri.imitrob_nlp.modules.CrowModule import CrowModule
import imitrob_hri.imitrob_nlp.modules.ColorDetector as ColorDetector
import imitrob_hri.imitrob_nlp.modules.LocationDetector as LocationDetector
from imitrob_hri.imitrob_nlp.structures.tagging.TaggedText import TaggedText
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager

from imitrob_hri.merging_modalities.probs_vector import ProbsVector

from operator import not_
def do_nothing(x):
    return x

class ObjectsDetectedData(object):
    ''' Object Detector result,
        Input for Object Grounder
    '''
    def __init__(self):
        self.objs_mentioned_cls = ProbsVector(c='default')
        self.objs_properties = {'color': ProbsVector(c='default')}
        self.objs_flags = {'to': []}

    @property
    def empty(self):
        if len(self.objs_mentioned_cls.names) == 0:
            return True
        else:
            return False
    
    def __str__(self):
        prop_str = "\n".join([f"{key}:\n{self.objs_properties[key]}" for key in self.objs_properties.keys()])
        return f"[_1_ Detected data] 1) objs_mentioned_cls:\n{self.objs_mentioned_cls}\n 2) objs_properties:\n{prop_str}"

    def keep_only_target_objects(self):
        self.keep_only_type(fun=not_)

    def keep_only_target_storages(self):
        self.keep_only_type(fun=do_nothing)

    def keep_only_type(self, fun):
        assert len(self.objs_mentioned_cls.names) == len(self.objs_mentioned_cls.p)
        # for key in self.objs_properties.keys():
        #     assert len(self.objs_mentioned_cls.names) == len(self.objs_properties[key].names), f"key: {key}, names: {self.objs_mentioned_cls.names}, property names: {self.objs_properties[key].names}"
        assert len(self.objs_mentioned_cls.names) == len(self.objs_flags['to'])
        
        for n in range(len(self.objs_flags['to'])-1, -1, -1):
            if fun(self.objs_flags['to'][n]):
                print(f"DELETING: {self.objs_mentioned_cls.names[n]}")
                self.objs_mentioned_cls.pop(n)
                # for key in self.objs_properties.keys():
                #     self.objs_properties[key].pop(n)
                for key in self.objs_flags.keys():
                    self.objs_flags[key].pop(n)
                

        for i in self.objs_flags['to']:
            assert fun(i) == False, f"objs flags: {self.objs_flags['to']}, self.objs_mentioned_cls: {self.objs_mentioned_cls}"
        assert len(self.objs_mentioned_cls.names) == len(self.objs_mentioned_cls.p)
        # for key in self.objs_properties.keys():
        #     assert len(self.objs_mentioned_cls.names) == len(self.objs_properties[key].names)
        assert len(self.objs_mentioned_cls.names) == len(self.objs_flags['to'])


# ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
# CROW = Namespace(f"{ONTO_IRI}#")
#db = Database()
# with db.onto as onto:
class ObjectDetector(CrowModule):
    """
    Detects an object in text.
    """
    # namespace = db.onto
    def __init__(self,language = 'en', client = None):
        self.logger = logging.getLogger(__name__)
        self.lang = language
        self.crowracle = client
        self.onto = self.crowracle.onto
        # self.db_api = DatabaseAPI()
        self.ui = UserInputManager(language = language)
        qres = self.crowracle.getTangibleObjectClasses()
        self.class_map = {}

        for c in qres:
            c_nlp = self.crowracle.get_nlp_from_uri(c)[0]
            self.class_map[c_nlp] = c
        # self.class_map = {
        #     "screwdriver": self.onto.Screwdriver,
        #     "hammer": self.onto.Hammer,
        #     "pliers": self.onto.Pliers,
        #     "glue" : self.onto.Glue,
        #     "panel" : self.onto.Panel,
        #     "cube" : self.onto.Cube
        # }
        self.templ_det = self.ui.load_file('templates_detection.json')
        self.obj_det_file = self.ui.load_file('objects_detection.json')
        self.guidance_file = self.ui.load_file('guidance_dialogue.json')
        self.synonyms_file = self.ui.load_file('synonyms.json')
        self.prepositions_file = self.ui.load_file('prepositions.json')
        self.verbs_file = self.ui.load_file('verbs.json')

    def detect_object(self, tagged_text : TaggedText, silent_fail=False):
        """Detects an object mentioned in the input text, extracts its properties and saves it
        in the object placeholder.

        Args:
            tagged_text (TaggedText): An input text
            silent_fail (bool, optional): _description_. Defaults to False.

        Returns:
            ObjectsDetectedData(): An object placeholder to be grounded later
        OR  None: No Object detected

        Current running pipeline:        
        process_sentence_callback()
        └── process_text()
            └── process_node()
                └── nlp_match()
                    └── detect_object()
                └── nlp_ground()
        """
        obj = ObjectsDetectedData()
        text = tagged_text.get_text()



        # try to detect one of the known objects in text
        for obj_str in self.class_map.keys():
            #TODO should be only NN, but we have a problem that kostka is detected as VB/VBD
            try:
                obj_str_lang = self.obj_det_file[self.lang][obj_str]
            except KeyError:
                continue
                # raise Exception(f"obj_str: {obj_str} not in obj_det_file")




            # if tagged_text.contains_pos_token(obj_str_lang, "NN") or tagged_text.contains_pos_token(
            #         obj_str_lang, "VBD") or tagged_text.contains_pos_token(obj_str_lang,
            #                                                                    "VB") or tagged_text.contains_pos_token(
            #         obj_str_lang, "NNS"):

            if tagged_text.contains_text(obj_str_lang):

                self.logger.debug(f"Object detected for \"{text}\": {obj}")
                self.ui.buffered_say(self.guidance_file[self.lang]["object_matched"] + text)
                obj_pv, objs_properties = self.detect_semantically_explicit_object(tagged_text, obj_str)
                if obj_pv.names[0] not in obj.objs_mentioned_cls.names:
                    obj.objs_mentioned_cls.add(obj_pv)
                    obj.objs_flags['to'].append(self.is_target_object(obj_str_lang, tagged_text))
                    obj.objs_properties['color'].add(objs_properties['color'])
            
            for obj_str_lang_syn in self.synonyms_file[self.lang][obj_str_lang]:
                if tagged_text.contains_text(obj_str_lang_syn):
                # if tagged_text.contains_pos_token(obj_str_lang_syn, "NN") or tagged_text.contains_pos_token(obj_str_lang_syn, "VBD") or tagged_text.contains_pos_token(obj_str_lang_syn, "VB") or tagged_text.contains_pos_token(obj_str_lang_syn, "NNS") :
                    obj_pv, objs_properties = self.detect_semantically_explicit_object(tagged_text, obj_str)
                    if obj_pv.names[0] not in obj.objs_mentioned_cls.names:
                        obj.objs_mentioned_cls.add(obj_pv)
                        obj.objs_flags['to'].append(self.is_target_object(obj_str_lang, tagged_text))
                        obj.objs_properties['color'].add(objs_properties['color'])
                
        # Handle cases like "it"
        # try to detect a coreference to an object
        if obj.empty:
            if tagged_text.contains_text(self.obj_det_file[self.lang]["it"]):
                obj = self.detect_coreferenced_object()
            else:
                if not silent_fail:
                    self.ui.buffered_say(self.guidance_file[self.lang]["object_not_matched"] + " " + text , say = 2)
                    self.ui.buffered_say(self.guidance_file[self.lang]["object_not_matched_repeat"], say = 3)
        
        if obj.empty: return None
        return obj

    def detect_semantically_explicit_object(self, tagged_text, obj_str):
        """Detect an object which is mentioned explicitly.

        Current running pipeline:        
        process_sentence_callback()
        └── process_text()
            └── process_node()
                └── nlp_match()
                    └── detect_object()
                        └── detect_semantically_explicit_object()
                └── nlp_ground()
        """
        cls = self.class_map[obj_str]
        
        # obj.is_a.append(cls)
        obj_str_lang = self.obj_det_file[self.lang][obj_str]

        # if tagged_text.contains_text(self.templ_det[self.lang]['any'] + " " + obj_str_lang):
        #     # the "any" flag will be used to select any object without asking the user
        #     obj.flags.append("any")
        # else:
        #     try:
        #         for obj_str_lang_syn in self.synonyms_file[self.lang][obj_str_lang]:
        #             if tagged_text.contains_text(self.templ_det[self.lang]['any']+" " + obj_str_lang_syn):
        #             # the "any" flag will be used to select any object without asking the user
        #                 obj.flags.append("any")
        #     except:
        #         pass

        obj_pv = ProbsVector(c='default')

        assert isinstance(tagged_text.tokens, list), tagged_text.tokens
        obj_pv.add(name=obj_str, p=0.0)
        print(obj_pv.names, obj_pv.p)
        # obj.objs_mentioned_cls.names = [obj_str_lang]
        # obj.objs_mentioned_cls.p = np.zeros((len(obj.objs_mentioned_cls.names)))

        if tagged_text.contains_text(obj_str_lang):
            idx = tagged_text.indices_of(obj_str_lang)
            # idx = obj.objs_mentioned_cls.names.index(obj_str_lang)
            # obj.objs_mentioned_cls.p[idx] = 1.0
            obj_pv.p = [1.0]
        else:
            #try:
            for obj_str_lang_syn in self.synonyms_file[self.lang][obj_str_lang]:
                if tagged_text.contains_text(obj_str_lang_syn):
                    idx = tagged_text.indices_of(obj_str_lang_syn)
                    # idx = obj.objs_mentioned_cls.names.index(obj_str_lang_syn)
                    # obj.objs_mentioned_cls.p[idx] = 0.99
                    obj_pv.p = [0.99]
            # except:
            #     pass
        # print(idx)
        # print(tagged_text)
        # print(tagged_text[0:idx])
            
        # seber cervenou kostku a modrou -> cervena 
        # seber kostku cervenou -> x
        # TODO: Check for how we want it
        min_idx = 0
        for idx_ in range(idx[0]):
            if self.is_VB(tagged_text.tokens[idx_]):
                min_idx = idx_
        tagged_text_color = tagged_text.cut(min_idx,idx[0])
        colors = self.detect_object_color(tagged_text_color)
        colors = list(set(colors))
        #self.detect_object_id(obj, tagged_text)
        #self.detect_object_location(obj, obj_str, tagged_text)

        objs_properties = {}
        objs_properties['color'] = ProbsVector(c='default')
        if len(colors) > 0:
            objs_properties['color'] = ProbsVector(template_names=colors, p=np.ones((len(colors))), c='default')
            

        return obj_pv, objs_properties

    # def detect_known_object(self, tagged_text, obj_str):
    #     cls = self.class_map[obj_str]
    #     obj = db.onto.ObjectPlaceholder()
    #     obj.is_a.append(cls)
    #
    #     if tagged_text.contains_text("any " + obj_str):
    #         obj.flags.append("any")
    #
    #     self.detect_object_color(obj, tagged_text)
    #     self.detect_object_location(obj, obj_str, tagged_text)
    #
    #     return obj

    def detect_coreferenced_object(self):
        """
        Detect that the text is referencing an object mentioned earlier.
        """

        obj = ObjectsDetectedData()
        obj.flags.append("last_mentioned")

        return obj

    def detect_object_location(self, obj, obj_str, tagged_text):
        # cut the part of the text that is sent into the location detector to avoid infinite loop
        # TODO: all of this is only a temporary solution, not intended to be used in the final product
        end_index = tagged_text.get_text().find(obj_str) + len(obj_str)
        new_tagged_text = tagged_text.cut(end_index, None)

        ld = LocationDetector.LocationDetector(language = self.lang)
        location = ld.detect_location(new_tagged_text)
        if location:
            obj.location = location

    def detect_object_color(self, tagged_text):
        cd = ColorDetector.ColorDetector(language = self.lang, client = self.crowracle)
        colors = cd.detect_color(tagged_text)
        
        return colors

    # def detect_object_id(self, obj, tagged_text):
    #     idet = IdDetector.IdDetector()
    #
    #     id = idet.detect_id(tagged_text)
    #
    #     if id is not None:
    #         obj.aruco_id = id

    def is_PP(self, tag):
        ''' Is preposition? 
            This is temporary function: tagging is not working properly
        '''
        # print("4  ", tag,  self.prepositions_file[self.lang].values())
        # print("5  ", tag in self.prepositions_file[self.lang].values())
        if tag in self.prepositions_file[self.lang].values():
            return True
        else:
            return False

    def is_VB(self, tag):
        ''' Is verb? 
            This is temporary function: tagging is not working properly 
        '''
        # print("4  ", tag,  self.verbs_file[self.lang].values())
        # print("5  ", tag in self.verbs_file[self.lang].values())
        if tag in self.verbs_file[self.lang].values():
            return True
        else:
            return False

    def is_target_object(self, to, tagged_text):
        ''' is target_object or target storage (target of manipulation)
            Checks whether before it is no preposition 
        '''

        for tag_idx in range(tagged_text.indices_of(to)[0]-1, -1, -1):
            # print("1  ", tagged_text.indices_of(to))
            # print("2  ", tagged_text.tokens[tag_idx])
            # print("3  ", self.is_PP(tagged_text.tokens[tag_idx]))
            if self.is_PP(tagged_text.tokens[tag_idx]):
                if tag_idx == 0:
                    return False
                if tag_idx-1>=0 and (not self.is_VB(tagged_text.tokens[tag_idx-1])):
                    return False
            if self.is_VB(tagged_text.tokens[tag_idx]):
                return True
        return True




