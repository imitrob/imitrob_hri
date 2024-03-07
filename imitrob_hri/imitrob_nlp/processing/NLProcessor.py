#!/usr/bin/env python3

#from nlp_crow.database.Database import Database, State
#from nlp_crow.database.DatabaseAPI import DatabaseAPI
from imitrob_hri.imitrob_nlp.database.Ontology import RobotProgram, RobotProgramOperator, RobotProgramOperand, RobotCustomProgram, Template
from imitrob_hri.imitrob_nlp.modules.GrammarParser import GrammarParser
from imitrob_hri.imitrob_nlp.modules.TemplateDetector import TemplateDetector
from imitrob_hri.imitrob_nlp.structures.tagging.ParsedText import ParseTreeNode
from imitrob_hri.imitrob_nlp.TemplateFactory import TemplateFactory, TemplateType

from imitrob_templates.templates import BaseTask

import logging
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager
from copy import deepcopy
import numpy as np
from imitrob_hri.merging_modalities.probs_vector import ProbsVector

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
        """Turns an input text into a program template which can be used for creating instructions for the robot
        (after grounding).
        The program template is dependent only on the content of the sentence and not
        on the current state of the workspace.

        Args:
            sentence (str): an input sentence

        Returns:
            RobotProgram():  - formalized instructions for the robot with placeholders for real objects and locations
        (right now, the behavior is undefined in case the sentence does not allow creating a valid program)

        Current running pipeline:        
        process_sentence_callback()
        └── process_text()
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
        """Process subnodes, individual sentences

        Args:
            subnode (ParseTreeNode): _description_

        Returns:
            RobotProgramOperand: _description_
        
        Current running pipeline:        
        process_sentence_callback()
        └── process_text()
            └── process_node()
        """        
        node = RobotProgramOperand()
        node.parsed_text = subnode
        # working with flat tagged text (without any tree structure)
        tagged_text = subnode.flatten()

        # using TemplateDetector to get a list of templates sorted by probability
        template_types = self.td.detect_templates(tagged_text)

        self.logger.debug(f"Templates detected for \"{tagged_text.get_text()}\": {[t.name for t in template_types]}")

        template_probs = ProbsVector(np.array([1.0] * len(template_types)), [t.name for t in template_types], c='default')
        template_o = []

        _1_detected_data = BaseTask.nlp_match(tagged_text, language = self.lang, client = self.crowracle)
        _2_grounded_data = BaseTask.nlp_ground(_1_detected_data, language=self.lang, client = self.crowracle)

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
            template._1_detected_data = _1_detected_data
            template._2_grounded_data = _2_grounded_data
            # equally detected templates, all have prob = 1.0
            
            # try to match all the template parameters
            
            template_o.append(deepcopy(template))
            # check if the template is matched successfully
            if template.grounded_data_filled(): # Found grounded match
                template_probs.p[template_probs.template_names.index(template_type.name)] = 1.0 
            else:                
                template_probs.p[template_probs.template_names.index(template_type.name)] = 1.0  # FIXME: should be lower, maybe (TDB)
                
        # more templates that gounded_data_filled, pick first
        final_template = None

        def penalize(final_template):
            ''' Need to be REVISED
            '''
            for req in ['target_action', 'target_object', 'target_storage']:
                if req in final_template.parameters:
                    penalization = 1.0 # No penalization
                else: 
                    penalization = 0.5
                    if req == 'target_object':
                        if 'to' in final_template._1_detected_data:
                            final_template._1_detected_data['to'].objs_mentioned_cls.p *= penalization
                            final_template._2_grounded_data['to'].to.p *= penalization
                    elif req == 'target_storage':
                        if 'ts' in final_template._1_detected_data:
                            final_template._1_detected_data['ts'].objs_mentioned_cls.p *= penalization
            return final_template

        for template_ in template_o:
            if template_.grounded_data_filled():
                final_template = template_
                final_template = penalize(final_template)
                break
        
        # No template has properly grounded data filled, give penalization 
        if final_template is None and len(template_types)>0: 
            # template.template_probs.p = 0.5 * template.template_probs.p
            final_template = template_o[0]

        # No template detected whatsoever            
        if final_template is None:
            final_template = self.tf.get_template_class_from_str('noop')()
            final_template.template_probs = ProbsVector(np.array([0.5]), ['noop'], c='default')
            # try to match all the template parameters
            final_template._1_detected_data = _1_detected_data
            final_template._2_grounded_data = _2_grounded_data

        final_template.template_probs = template_probs

        # save the filled template in the program node
        node.template = final_template
        return node
