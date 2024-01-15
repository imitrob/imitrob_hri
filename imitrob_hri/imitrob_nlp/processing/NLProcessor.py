#!/usr/bin/env python3
"""
Copyright (c) 2019 CIIRC, CTU  in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner
"""

#from nlp_crow.database.Database import Database, State
#from nlp_crow.database.DatabaseAPI import DatabaseAPI
from imitrob_hri.imitrob_nlp.database.Ontology import RobotProgram, RobotProgramOperator, RobotProgramOperand, RobotCustomProgram, Template
from imitrob_hri.imitrob_nlp.modules.GrammarParser import GrammarParser
from imitrob_hri.imitrob_nlp.modules.TemplateDetector import TemplateDetector
from imitrob_hri.imitrob_nlp.structures.tagging.ParsedText import ParseTreeNode
from imitrob_hri.imitrob_nlp.TemplateFactory import TemplateFactory, TemplateType

import logging
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager

class NLProcessor():
    def __init__(self,language="en", client = None):
        self.gp = GrammarParser(language = language)
        self.td = TemplateDetector(language=language)
        self.tf = TemplateFactory()
        self.crowracle = client
        self.lang = language
        self.ui = UserInputManager(language = language)

        self.logger = logging.getLogger(__name__)
        self.guidance_file = self.ui.load_file('guidance_dialogue.json')


    def process_text(self, sentence : str):
        """
        Turns an input text into a program template which can be used for creating instructions for the robot
        (after grounding).

        The program template is dependent only on the content of the sentence and not
        on the current state of the workspace.

        Parameters
        ----------
        sentence  an input sentence as string

        Returns
        -------
        a program template - formalized instructions for the robot with placeholders for real objects and locations
        (right now, the behavior is undefined in case the sentence does not allow creating a valid program)
        """
        parsed_text = self.gp.parse(sentence)
        root = parsed_text.parse_tree
        # db_api = DatabaseAPI()
        #state = db_api.get_state()

        # if state == State.LEARN_FROM_INSTRUCTIONS:
        #     program = RobotCustomProgram()
        # else:
        #     program = RobotProgram()

        # hardcoded program structure: all subprograms are located directly under the root node "AND"
        program = RobotProgram()
        program.root = RobotProgramOperator(operator_type="AND")
        for subnode in root.subnodes:
            if type(subnode) is ParseTreeNode:
                # create a single robot instructon
                program_node = self.process_node(subnode)
                program.root.add_child(program_node)

        return program

    def process_node(self, subnode : ParseTreeNode) -> RobotProgramOperand:
        node = RobotProgramOperand()
        node.parsed_text = subnode
        # working with flat tagged text (without any tree structure)
        tagged_text = subnode.flatten()

        # using TemplateDetector to get a list of templates sorted by probability
        template_types = self.td.detect_templates(tagged_text)

        self.logger.debug(f"Templates detected for \"{tagged_text.get_text()}\": {[t.name for t in template_types]}")

        # try to sequentially match each template
        for template_type in template_types:
            # custom template cannot be parametrized yet -> no matching required
            # TODO the condition looks kind of stupid
            if type(template_type) is not TemplateType:
                # custom template is already a valid program
                node = template_type.root
                template = None
                break

            # get an object representing the template
            template = self.tf.get_template(template_type) # type: Template

            # try to match all the template parameters
            template.match_tagged_text(tagged_text, language = self.lang, client = self.crowracle)
            template.ground(client = self.crowracle)
            # check if the template is matched successfully
            if template.is_filled():
                break
        else:
            template = None

        # save the filled template in the program node
        node.template = template
        return node
