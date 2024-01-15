#!/usr/bin/env python
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner
"""
import json
import os
from typing import List

from pkg_resources import resource_filename

#from crow_nlp.nlp_crow.database.DatabaseAPI import DatabaseAPI
from imitrob_hri.imitrob_nlp.modules.CrowModule import CrowModule
#from crow_nlp.nlp_crow.structures.tagging.MorphCategory import POS
#from crow_nlp.nlp_crow.structures.tagging.ParsedText import TaggedText
#from crow_nlp.nlp_crow.structures.tagging.Tag import Tag
from imitrob_hri.imitrob_nlp.TemplateFactory import TemplateType as tt
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager
import logging
import rclpy

#db_api = DatabaseAPI()
#db = db_api.get_db()
# onto = db_api.get_onto()

# with onto:
class TemplateDetector(CrowModule):
    """
    First part of the NL pipeline. Preliminary template detection in the text.
    """
  #  namespace = db.onto
    def __init__(self,language = 'en'):
        self.lang = language
        self.logger = logging.getLogger(__name__)

        self.ui = UserInputManager(language = language)

        self.templ_det = self.ui.load_file('templates_detection.json')
        self.guidance_file = self.ui.load_file('guidance_dialogue.json')

    def detect_templates(self, tagged_text) -> List[tt]:
        """
        Tries to guess which template should be used to represent robot instructions for the chunk of
        text. Can detect more templates, but only the first one which matches will be used later.

        Parameters
        ----------
        tagged_text  a tagged text in which the template should be detected

        Returns
        -------
        a list of guessed templates for the tagged text sorted by their probability
        """
        templates = []
        detect_fns = [self.detect_pick,
                      self.detect_point
                        # self.detect_define_storage,
                        # self.detect_define_position,
                        # self.detect_remove_command_x,
                        # self.detect_remove_command_last,
                        # self.detect_fetch,
                        # self.detect_fetch_to,
                        # self.detect_release,
                        #   # self.detect_apply_glue,
                        #   # self.detect_put, #exchanged for fetch_to
                        # self.detect_tidy,
                        # self.detect_stop,
                        # self.detect_build,
                        # self.detect_cancel_build,
                        # self.detect_remove_product,
                          # self.detect_learn,
                          # self.detect_tower,
                          # self.detect_demonstration_list,
                          # self.detect_define_area
                    ]

        # try to find custom templates (compound actions) first
        # custom_templates = self.detect_custom_templates(tagged_text)

        # if custom_templates:
        #     templates += custom_templates

        # add detected basic templates (actions)
        template_found = False
        for detect_fn in detect_fns:
            res = detect_fn(tagged_text)

            if res:
                template_found = True
                templates += res
        if not template_found:
            self.logger.error("No template match for \"{}\"".format(tagged_text.get_text()))
            self.ui.buffered_say(self.guidance_file[self.lang]["no_template_match"] + tagged_text.get_text(), say = 2)

        # print(templates)
        return templates


    # def detect_custom_templates(self, tagged_text : TaggedText):
    #     """
    #     Retrieves learned custom templates (compound actions) from the database
    #     and tries to detect them in the text.
    #
    #     Parameters
    #     ----------
    #     tagged_text  a tagged text in which a custom template should be detected
    #
    #     Returns
    #     -------
    #     a list of custom templates detected in the text, an empty list if no template is detected
    #     """
    #     all_custom_templates = db_api.get_custom_templates()
    #
    #     custom_templates = []
    #
    #     for custom_template in all_custom_templates:
    #         custom_template_name = custom_template.name[1:]
    #
    #         if custom_template_name.lower() in tagged_text.get_text().lower():
    #             custom_templates.append(custom_template)
    #     return custom_templates


    def detect_pick(self, tagged_text) -> List[tt]:
        """
        Detector for PickTask
        """
        # if tagged_text.contains_pos_token(self.templ_det[self.lang]['take'], "VB") or tagged_text.contains_pos_token(self.templ_det[self.lang]['pick'], "VB"):
        #     self.ui.say(self.guidance_file[self.lang]["template_match"]+self.templ_det[self.lang]['pick'])

            #if tagged_text.contains_pos_token("take", "VB") or \
        #        tagged_text.contains_pos_token("pick", "VB"):
        if tagged_text.contains_text(self.templ_det[self.lang]['pick']):
            self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['pick'])
            return [tt.PICK_TASK]
        
    def detect_point(self, tagged_text) -> List[tt]:
        """
        Detector for PointTask
        """
        #if tagged_text.contains_text("point"):
        if tagged_text.contains_text(self.templ_det[self.lang]['point']):
            self.ui.buffered_say(self.guidance_file[self.lang]["template_match"]+ self.templ_det[self.lang]['point'])
            return [tt.POINT_TASK]


    # def detect_apply_glue(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for ApplyGlueTask
    #     """
    #     if tagged_text.contains_pos_token(self.templ_det[self.lang]['glue'], "VB") or tagged_text.contains_pos_token(self.templ_det[self.lang]['glue'], "NNS")  or tagged_text.contains_pos_token(self.templ_det[self.lang]['glue'], "NNP"):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['glue'])
    #         return [tt.APPLY_GLUE]

    # def detect_learn(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for LearnNewTask
    #     """
    #     if tagged_text.contains_text("learn") and tagged_text.contains_text("new task"):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['learn_new_task'])
    #         return [tt.LEARN_NEW_TASK]

    # def detect_put(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for PutTask
    #     """
    #     #if tagged_text.contains_text("put"):
    #     if tagged_text.contains_text(self.templ_det[self.lang]['put']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"]+ self.templ_det[self.lang]['put'])
    #         return [tt.PUT_TASK]


    # def detect_tidy(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for Tidy
    #     """
    #     #if tagged_text.contains_text("tidy"):
    #     if tagged_text.contains_text(self.templ_det[self.lang]['tidy']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"]+ self.templ_det[self.lang]['tidy'])
    #         return [tt.TIDY_TASK]

    # def detect_tower(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for LearnTowerFromDemonstration
    #     """
    #     if tagged_text.contains_text("learn") and tagged_text.contains_text("tower"):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"]+ self.templ_det[self.lang]['learn_new_tower'])
    #         return [tt.LEARN_TOWER]


    # def detect_demonstration_list(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for DemonstrationList
    #     """
    #     if tagged_text.contains_text("show") and tagged_text.contains_text("demonstration"):
    #         self.ui.say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['demonstration'])
    #         return [tt.DEMONSTRATION_LIST]


    # def detect_define_area(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for DefineArea
    #     """
    #     if tagged_text.contains_text("define") and tagged_text.contains_text("area"):
    #         self.ui.say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['define_area'])
    #         return [tt.DEFINE_AREA]

    # def detect_define_storage(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for DefineStorageAction
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['storage']) and tagged_text.contains_text(self.templ_det[self.lang]['define']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['storage'])
    #         return [tt.DEFINE_STORAGE]

    # def detect_define_position(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for DefinePositionAction
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['position']) and tagged_text.contains_text(self.templ_det[self.lang]['define']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['position'])
    #         return [tt.DEFINE_POSITION]

    # def detect_remove_command_last(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for RemoveCommandLast
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['remove_last_command']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['remove_last_command'])
    #         return [tt.REMOVE_COMMAND_LAST]

    # def detect_remove_command_x(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for RemoveCommandX
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['remove_command']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['remove_command'])
    #         return [tt.REMOVE_COMMAND_X]

    # def detect_fetch(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for Fetch
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['give']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['give'])
    #         return [tt.FETCH]

    # def detect_fetch_to(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for FetchTo
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['put']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['put'])
    #         return [tt.FETCH_TO]

    # def detect_release(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for Release
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['release']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['release'])
    #         return [tt.RELEASE]

    # def detect_stop(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for Stop
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['stop']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['stop'])
    #         return [tt.STOP]

    # def detect_build(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for action to build a given assembly
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['build']) and (tagged_text.contains_text(self.templ_det[self.lang]['cancel'])==False):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['build'])
    #         return [tt.BUILD]

    # def detect_cancel_build(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for action to build a given assembly
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['build']) and tagged_text.contains_text(self.templ_det[self.lang]['cancel']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['cancel_build'])
    #         return [tt.BUILD_CANCEL]

    # def detect_remove_product(self, tagged_text) -> List[tt]:
    #     """
    #     Detector for action to remove the product
    #     """
    #     if tagged_text.contains_text(self.templ_det[self.lang]['remove_product']):
    #         self.ui.buffered_say(self.guidance_file[self.lang]["template_match"] + self.templ_det[self.lang]['remove_product'])
    #         return [tt.PRODUCT_REMOVE]
