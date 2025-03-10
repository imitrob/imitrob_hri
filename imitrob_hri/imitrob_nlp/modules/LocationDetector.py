#!/usr/bin/env python
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner, Karla Štěpánová
@mail:  karla.stepanova@cvut.cz
"""
import logging

# from crow_nlp.nlp_crow.database.Database import Database
# from crow_nlp.nlp_crow.modules.AreaDetector import AreaDetector
from imitrob_hri.imitrob_nlp.modules.CrowModule import CrowModule
from imitrob_hri.imitrob_nlp.structures.tagging.TaggedText import TaggedText
from imitrob_hri.imitrob_nlp.utils import nlp
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager

# db = Database()

# with db.onto as onto:
class LocationDetector(CrowModule):
    """
    Detects a location in text.
    """
    # namespace = db.onto
    def __init__(self,language = 'en'):
        self.logger = logging.getLogger(__name__)
        self.lang = language
        self.ui = UserInputManager(language = self.lang)
        self.templ_det = self.ui.load_file('templates_detection.json')

    def detect_location(self, tagged_text: TaggedText):
        """
        Tries to extract the information about the location in the text.

        Parameters
        ----------
        tagged_text  a text in which the location should be detected

        Returns
        -------
        the location object or None if no location is detected
        """
        tagged_text.lang = self.lang
        loc = self.detect_absolute_location(tagged_text)

        if not loc:
            self.logger.debug(f"Absolute Location not detected for \"{tagged_text.get_text()}\": {loc}. Trying relative location")
            loc = self.detect_relative_location(tagged_text)

        if loc:
            self.logger.debug(f"Location detected for \"{tagged_text.get_text()}\": {loc}")
        else:
            self.logger.debug(f"Location not detected for \"{tagged_text.get_text()}\": {loc}")

        return loc

    def detect_absolute_location(self, tagged_text: TaggedText):
        if tagged_text.contains_text(self.templ_det[self.lang]["here"]):
            return self.detect_current_finger_location()

        elif tagged_text.contains_text(self.templ_det[self.lang]["down"]):
            return self.detect_current_robot_handle_location()

        elif tagged_text.contains_text(self.templ_det[self.lang]["table"]):
            loc = self.onto.Location()
            loc.loc_type = "table"
            loc.x = 0.410
            loc.y = 0.49
            loc.z = 0
            return loc

        return self.detect_location_from_text(tagged_text)


    # def detect_relative_location(self, tagged_text : TaggedText):
    #     # for preventing cyclic imports in previous Python versions
    #     import crow_nlp.nlp_crow.modules.ObjectDetector as ObjectDetector
    #
    #     loc = None
    #     obj_rel_locs = ["center", "left", "right", "top", "bottom"]
    #
    #     for rel_loc in obj_rel_locs:
    #         if tagged_text.contains_text(self.templ_det[self.lang][rel_loc]):
    #             # TODO temporary solution: detect the object which the location refers to in the part of the sentence *after* the location
    #             index = tagged_text.indices_of(rel_loc)[0]
    #             tagged_text_cut = tagged_text.cut(index + 1, None)
    #
    #             # detect to which object the location refers
    #             od = ObjectDetector.ObjectDetector()
    #             relative_to = od.detect_object(tagged_text_cut)
    #
    #             if not relative_to:
    #                 ad = AreaDetector()
    #                 relative_to = ad.detect_area(tagged_text_cut)
    #
    #             if relative_to:
    #                 loc = db.onto.RelativeLocation()
    #                 loc.loc_type = rel_loc
    #                 loc.relative_to = relative_to
    #
    #     return loc

    def detect_location_from_text(self, tagged_text):
        pos_lang = self.templ_det[self.lang]["position"]
        regex_pos_list = [
            #(rf"{pos_lang}", "NN"),
            (r"position", "NN"),
            (r".*", "CD"),
            (r".*", "CD")
        ]
        # looking for " (...) position/NN (...) X/CD (...) Y/CD (...)"
        res = tagged_text.match_regex_pos_list(regex_pos_list)

        if res:
            loc = self.onto.Location()
            loc.x = self.get_coordinate(res[1].string)
            loc.y = self.get_coordinate(res[2].string)

            return loc

    def detect_current_robot_handle_location(self):
        # TODO this should be changed to RelativeLocation and then grounded according to the current robot handle location
        return self.generate_random_location()

    def detect_current_finger_location(self):
        # TODO connect with the vision module
        return self.generate_random_location()

    def get_coordinate(self, text):
        try:
            return int(text)
        except ValueError:
            return nlp.text2int(text)

    # def generate_random_location(self):
    #     import random
    #     loc = db.onto.Location()
    #     loc.x = random.randint(0, 5)/10
    #     loc.y = random.randint(0, 5)/10
    #     return loc
