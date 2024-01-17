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

class ObjectsDetectedData(object):
    ''' Object Detector result,
        Input for Object Grounder
    '''
    def __init__(self):
        self.objs_mentioned_cls = ProbsVector(c='default')
        self.objs_properties = {'color': ProbsVector(c='default')}
        self.flags = []

    @property
    def empty(self):
        if len(self.objs_mentioned_cls.names) == 0:
            return True
        else:
            return False


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
           # print(self.crowracle.get_nlp_from_uri(c)[0])
            c_nlp = self.crowracle.get_nlp_from_uri(c)[0]
           # print(c_nlp)
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
            # try:
            # TODO we would like not to break after the first detection but detect even if more objects were mentioned
            #TODO should be only NN, but we have a problem that kostka is detected as VB/VBD
            try:
                obj_str_lang = self.obj_det_file[self.lang][obj_str]
            except KeyError:
                continue
            # if tagged_text.contains_pos_token(obj_str_lang, "NN") or tagged_text.contains_pos_token(
            #         obj_str_lang, "VBD") or tagged_text.contains_pos_token(obj_str_lang,
            #                                                                    "VB") or tagged_text.contains_pos_token(
            #         obj_str_lang, "NNS"):
            if tagged_text.contains_text(obj_str_lang):
                self.logger.debug(f"Object detected for \"{text}\": {obj}")
                self.ui.buffered_say(self.guidance_file[self.lang]["object_matched"] + text)
                obj_pv, objs_properties = self.detect_semantically_explicit_object(tagged_text, obj_str)
                obj.objs_mentioned_cls.add(obj_pv)
                obj.objs_properties['color'].add(objs_properties['color'])
                break
            #try:
            for obj_str_lang_syn in self.synonyms_file[self.lang][obj_str_lang]:
                if tagged_text.contains_text(obj_str_lang_syn):
                # if tagged_text.contains_pos_token(obj_str_lang_syn, "NN") or tagged_text.contains_pos_token(obj_str_lang_syn, "VBD") or tagged_text.contains_pos_token(obj_str_lang_syn, "VB") or tagged_text.contains_pos_token(obj_str_lang_syn, "NNS") :
                    obj_pv, objs_properties = self.detect_semantically_explicit_object(tagged_text, obj_str)
                    obj.objs_mentioned_cls.add(obj_pv)
                    obj.objs_properties['color'].add(objs_properties['color'])
                    break
            # except:
            #     pass
            # except:
            #     pass
                
        # Handle cases like "it"
        # try to detect a coreference to an object
        if obj.empty:
            if tagged_text.contains_text(self.obj_det_file[self.lang]["it"]):
                obj = self.detect_coreferenced_object()
            else:
                if not silent_fail:
                    self.ui.buffered_say(self.guidance_file[self.lang]["object_not_matched"] + " " + text , say = 2)
                    self.ui.buffered_say(self.guidance_file[self.lang]["object_not_matched_repeat"], say = 3)
        
        print(f" ** [Object Detector] ended with: **\n{obj.objs_mentioned_cls}\n************************************")

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
        tagged_text_color = tagged_text.cut(0,idx[0])
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