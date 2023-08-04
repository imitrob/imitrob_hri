#!/usr/bin/env python3
from crow_nlp.nlp_crow.database.Ontology import Template
from crow_nlp.nlp_crow.modules.ObjectDetector import ObjectDetector
from crow_nlp.nlp_crow.modules.ObjectGrounder import ObjectGrounder
from crow_nlp.nlp_crow.structures.tagging.TaggedText import TaggedText
from crow_nlp.nlp_crow.modules.UserInputManager import UserInputManager
from crow_msgs.msg import CommandType

import logging

class PickTask(Template):
    """
    A template for the pick task = a robot instruction representing picking a specific object.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lang = 'cs'
        # dialogue tools
        self.ui = UserInputManager(language = self.lang)
        # template detector - simplified view: seaches for a word that matches template
        self.templ_det = self.ui.load_file('templates_detection.json')
        self.parameters = ['action', 'action_type', 'target', 'target_type']
        self.target = [] # object to pick
        self.target_type = 'onto_uri'
        self.action_type = self.templ_det[self.lang]['pick']
        self.action = CommandType.PICK
        
        
        ###TODO: replaced by default robot behaviour (pick and home?)
        self.location = [0.53381, 0.18881, 0.22759]  # temporary "robot default" position
        self.location_type = 'xyz'
        
        self.logger = logging.getLogger(__name__)

        # related to parameters ?
        self.compare_types = ['action', 'selection']

    def match(self, tagged_text : TaggedText, language = 'en', client = None) -> None:
        od = ObjectDetector(language = language, client = client)
        self.target = od.detect_object(tagged_text)

    def evaluate(self, language = 'en', client = None) -> None:

        self.lang = language
        self.ui = UserInputManager(language=self.lang)
        self.guidance_file = self.ui.load_file('guidance_dialogue.json')
        og = ObjectGrounder(language=self.lang, client=client)
        if self.target:
            self.target, self.target_ph_cls, self.target_ph_color, self.target_ph_loc = og.ground_object(obj_placeholder=self.target)
            names_to_add = ['target_ph_cls', 'target_ph_color', 'target_ph_loc']
            for name in names_to_add:
                if getattr(self, name):
                    self.parameters.append(name)

    def execute(self) -> None:
        self.target.location.x
        self.target.location.y
        print(self)

    def has_compare_type(self, compare_type):
        if compare_type in self.compare_types:
            return True
        else:
            return False
    
