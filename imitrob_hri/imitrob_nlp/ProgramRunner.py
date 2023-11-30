#!/usr/bin/env python
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner, Karla Štěpánová
@mail:  karla.stepanova@cvut.cz
"""
# from nlp_crow.database.Database import Database
# from nlp_crow.database.DatabaseAPI import DatabaseAPI
# from crow_nlp.nlp_crow.modules.LocationGrounder2 import LocationGrounder
from crow_nlp.nlp_crow.modules.ObjectGrounder import ObjectGrounder
from crow_nlp.nlp_crow.database.Ontology import RobotProgram, RobotProgramOperator, RobotProgramOperand, RobotCustomProgram, Template

import owlready2 as ow
import logging

from imitrob_hri.imitrob_nlp.TemplateFactory import TemplateType

# db = Database()

# with db.onto as onto:
class ProgramRunner:
    """
    Grounds object and location placeholders in the template so that they refer
    to real objects and locations in the workspace. The result can be used to
    create instructions for the robot.
    """
    def __init__(self, language = 'en', client = None):
        self.logger = logging.getLogger(__name__)
        self.lang = language
        self.crowracle = client
        self.og = ObjectGrounder(language = self.lang, client = self.crowracle)
        # self.lg = LocationGrounder()
        # self.db_api = DatabaseAPI()

    def evaluate(self, program : RobotProgram):
        self.evaluate_recursive(program.root)

        return program

    def evaluate_recursive(self, node) -> None:
        if type(node) == RobotProgramOperator:
            self.evaluate_operator(node)

        elif type(node) == RobotProgramOperand:
            self.evaluate_operand(node)


    def evaluate_operator(self, node: RobotProgramOperator):
        for child in node.children:
            self.evaluate_recursive(child)

    def evaluate_operand(self, node: RobotProgramOperand):
        template = node.template

        if template is None:
            return

        # call the evaluate() method for each template
        template.evaluate(language = self.lang, client = self.crowracle)

        for param, input in template.get_inputs().items():
            if not input:
                self.logger.error(f"Could not fill {param}")