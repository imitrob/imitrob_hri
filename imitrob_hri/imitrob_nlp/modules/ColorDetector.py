#!/usr/bin/env python
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner, Karla Štěpánová
@mail:  karla.stepanova@cvut.cz
"""
from imitrob_hri.imitrob_nlp.modules.CrowModule import CrowModule
from imitrob_hri.imitrob_nlp.structures.tagging.TaggedText import TaggedText
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager
# from crow_ontology.crowracle_client import CrowtologyClient
# from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD


# ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
# CROW = Namespace(f"{ONTO_IRI}#")


class ColorDetector(CrowModule):
    """
    Detects colors in text.
    """
    def __init__(self,language = 'en', client = None):
        self.crowracle = client
        #self.colors = [self.crowracle.get_nlp_from_uri(c) for c in self.crowracle.getColors()] #<class 'list'>: ['black', 'blue', 'cyan', 'purple', 'dark purple', 'gold', 'green', 'blue', 'light blue', 'magenta', 'red', 'dark red', 'red wine', 'white', 'yellow']
        #TODO fixed just temporary for Fuseki, need to use the above command to quer crowracle, now it gives specific types of colors within a list
        # [['gold'], ['blue'], ['dark purple', 'purple'], ['red'], ['green'], ['white'], ['blue', 'light blue'], ['magenta'], ['red wine', 'dark red'], ['black'], ['yellow'], ['cyan']]
        self.colors = ['gold', 'blue', 'purple', 'red', 'green', 'white', 'blue', 'magenta', 'red wine', 'black', 'yellow', 'cyan']
        self.lang = language
        self.ui = UserInputManager(language = self.lang)
        self.templ_det = self.ui.load_file('templates_detection.json')
        self.synonym_file = self.ui.load_file('synonyms.json')
        self.guidance_file = self.ui.load_file('guidance_dialogue.json')

    def detect_color(self, text : TaggedText):
        """
        A simple method for detecting a color in text.
        Yes, there is some space for improvement.

        Parameters
        ----------
        text  an input text
        """
        
        mentioned_colors = []
        for color in self.colors:
            try:
                color_lang = self.templ_det[self.lang][color]
                if color_lang in text.get_text():
                    self.ui.buffered_say(self.guidance_file[self.lang]["color_matched"] + color_lang)  # f"{cls}")
                    mentioned_colors.append(color)
                try:
                    for color_lang_syn in self.synonym_file[self.lang][color_lang]:
                        if color_lang_syn in text.get_text():
                            self.ui.buffered_say(self.guidance_file[self.lang]["color_matched"] + color_lang_syn)
                            mentioned_colors.append(color)

                except:
                    pass
            except:
                pass

        return mentioned_colors