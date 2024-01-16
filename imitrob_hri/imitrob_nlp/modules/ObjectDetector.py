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
from imitrob_hri.imitrob_nlp.database.Ontology import ObjectsDetectedData
import imitrob_hri.imitrob_nlp.modules.LocationDetector as LocationDetector
from imitrob_hri.imitrob_nlp.structures.tagging.TaggedText import TaggedText
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager



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
        """
        Detects an object mentioned in the input text, extracts its properties and saves it
        in the object placeholder.

        Parameters
        ----------
        tagged_text  an input text

        Returns
        -------
        an object placeholder to be grounded later
        """
        obj = None
        text = tagged_text.get_text()

        # try to detect one of the known objects in text
        for obj_str in self.class_map.keys():
            try:
                # TODO we would like not to break after the first detection but detect even if more objects were mentioned
                #TODO should be only NN, but we have a problem that kostka is detected as VB/VBD
                obj_str_lang = self.obj_det_file[self.lang][obj_str]
                # if tagged_text.contains_pos_token(obj_str_lang, "NN") or tagged_text.contains_pos_token(
                #         obj_str_lang, "VBD") or tagged_text.contains_pos_token(obj_str_lang,
                #                                                                    "VB") or tagged_text.contains_pos_token(
                #         obj_str_lang, "NNS"):
                if tagged_text.contains_text(obj_str_lang):
                    self.logger.debug(f"Object detected for \"{text}\": {obj}")
                    self.ui.buffered_say(self.guidance_file[self.lang]["object_matched"] + text)
                    obj = self.detect_explicit_object(tagged_text, obj_str)
                    break
                try:
                    for obj_str_lang_syn in self.synonyms_file[self.lang][obj_str_lang]:
                        if tagged_text.contains_text(obj_str_lang_syn):
                        # if tagged_text.contains_pos_token(obj_str_lang_syn, "NN") or tagged_text.contains_pos_token(obj_str_lang_syn, "VBD") or tagged_text.contains_pos_token(obj_str_lang_syn, "VB") or tagged_text.contains_pos_token(obj_str_lang_syn, "NNS") :
                            obj = self.detect_explicit_object(tagged_text, obj_str)
                            break
                except:
                    pass
            except:
                pass
        # try to detect a coreference to an object
        if obj is None:
            if tagged_text.contains_text(self.obj_det_file[self.lang]["it"]):
                obj = self.detect_coreferenced_object()
            else:
                if not silent_fail:
                    self.ui.buffered_say(self.guidance_file[self.lang]["object_not_matched"] + " " + text , say = 2)
                    self.ui.buffered_say(self.guidance_file[self.lang]["object_not_matched_repeat"], say = 3)

        return obj

    def detect_explicit_object(self, tagged_text, obj_str):
        """
        Detect an object which is mentioned explicitly.
        """
        cls = self.class_map[obj_str]
        obj = ObjectsDetectedData()
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

        # object_names = obj_str_lang
        obj.objs_mentioned_cls = obj_str
        obj.objs_mentioned_cls_probs = np.zeros((len(obj.objs_mentioned_cls)))

        object_properties = {}

        if tagged_text.contains_text(obj_str_lang):
            idx = tagged_text.indices_of(obj_str_lang)
            obj.objs_mentioned_cls_probs[idx] = 1.0
        else:
            try:
                for obj_str_lang_syn in self.synonyms_file[self.lang][obj_str_lang]:
                    if tagged_text.contains_text(obj_str_lang_syn):
                        idx = tagged_text.indices_of(obj_str_lang_syn)
                        obj.objs_mentioned_cls_probs[idx] = 0.99
            except:
                pass
        # print(idx)
        # print(tagged_text)
        # print(tagged_text[0:idx])
            
        # seber cervenou kostku a modrou -> cervena 
        # seber kostku cervenou -> x
        # TODO: Check for how we want it
        tagged_text_color = tagged_text.cut(0,idx[0])
        colors = self.detect_object_color(tagged_text_color)
        #self.detect_object_id(obj, tagged_text)
        #self.detect_object_location(obj, obj_str, tagged_text)

        if len(colors) > 0:
            obj.objs_properties['color'] = {color: 1.0 for color in colors}

        return obj

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